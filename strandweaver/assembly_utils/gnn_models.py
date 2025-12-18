#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph Neural Network Models for Assembly Path Prediction.

This module provides production-ready GNN implementations for:
1. Edge Confidence Prediction: Classify edges as correct/incorrect
2. Path Scoring: Score multi-edge paths through graph
3. Node Classification: Classify nodes by haplotype/repeat status

Architecture:
- Message Passing GNN with graph convolution layers
- Attention mechanisms for edge importance weighting
- Multi-task learning (edge + node + path prediction)

Model variants:
- SimpleGNN: Lightweight, 2-3 layers
- MediumGNN: Balanced, 4-5 layers
- DeepGNN: Production, 6-8 layers with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAT, GATv2, MessagePassing
from torch_geometric.data import Data, DataLoader
import logging
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GNNConfig:
    """Configuration for GNN model."""
    # Architecture
    num_node_features: int = 12
    num_edge_features: int = 11
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    
    # Model
    use_attention: bool = True
    use_residual: bool = True
    use_batch_norm: bool = True
    output_dim: int = 1  # Binary classification: correct/incorrect edge
    
    # Device
    device: str = "cpu"


class EdgeConvLayer(MessagePassing):
    """
    Custom edge convolution layer with edge features.
    
    Learns edge importance from node context and edge features.
    """
    
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Node transformation
        self.lin_src = nn.Linear(in_channels, out_channels)
        self.lin_dst = nn.Linear(in_channels, out_channels)
        
        # Edge feature transformation
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        
        # Combined transformation
        self.lin_combined = nn.Linear(3 * out_channels, out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_channels)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, edge_dim)
        
        Returns:
            Updated node features (N, out_channels)
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute edge messages."""
        msg_src = self.lin_src(x_i)
        msg_dst = self.lin_dst(x_j)
        
        if edge_attr is not None:
            msg_edge = self.lin_edge(edge_attr)
            msg = torch.cat([msg_src, msg_dst, msg_edge], dim=-1)
        else:
            msg = torch.cat([msg_src, msg_dst, torch.zeros_like(msg_src)], dim=-1)
        
        msg = self.lin_combined(msg)
        return F.relu(msg)
    
    def aggregate(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Aggregate messages."""
        return aggr_out


class PathGNNModel(nn.Module):
    """
    Graph Neural Network for assembly path prediction.
    
    Multi-task architecture:
    1. Edge classification: Predict edge correctness
    2. Node classification: Classify node haplotype
    3. Path scoring: Score complete paths
    """
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info(f"Initializing {self.__class__.__name__} "
                   f"(layers={config.num_layers}, hidden={config.hidden_channels})")
        
        # Edge convolution layers
        self.edge_convs = nn.ModuleList()
        in_channels = config.num_node_features
        for i in range(config.num_layers):
            self.edge_convs.append(
                EdgeConvLayer(in_channels, config.hidden_channels, 
                            config.num_edge_features)
            )
            if config.use_batch_norm:
                self.edge_convs.append(nn.BatchNorm1d(config.hidden_channels))
            in_channels = config.hidden_channels
        
        # Attention layers (optional)
        if config.use_attention:
            self.attention = GATv2Conv(
                in_channels=config.hidden_channels,
                out_channels=config.hidden_channels // 2,
                heads=4,
                dropout=config.dropout
            )
            attention_out = config.hidden_channels * 2
        else:
            self.attention = None
            attention_out = config.hidden_channels
        
        # Task heads
        # 1. Edge classification head
        self.edge_head = nn.Sequential(
            nn.Linear(attention_out * 2, config.hidden_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, config.output_dim),
            nn.Sigmoid()
        )
        
        # 2. Node classification head
        self.node_head = nn.Sequential(
            nn.Linear(attention_out, config.hidden_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, 4),  # hap_a, hap_b, both, repeat, unknown
            nn.Softmax(dim=-1)
        )
        
        # 3. Path scoring head
        self.path_head = nn.Sequential(
            nn.Linear(attention_out * 3, config.hidden_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GNN.
        
        Args:
            data: PyG Data object with x (node features), edge_index, edge_attr
        
        Returns:
            Dict with:
            - edge_logits: Edge classification scores (E, 1)
            - node_logits: Node classification logits (N, 4)
            - path_scores: Path scores (if path_indices provided)
        """
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device) if data.edge_attr is not None else None
        
        # Pass through edge convolution layers
        for layer in self.edge_convs:
            if isinstance(layer, EdgeConvLayer):
                x = layer(x, edge_index, edge_attr)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
        
        # Apply attention if enabled
        if self.attention is not None:
            x_attn = self.attention(x, edge_index)
            x = torch.cat([x, x_attn], dim=-1)
        
        # Task 1: Edge classification
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_logits = self.edge_head(edge_features)
        
        # Task 2: Node classification
        node_logits = self.node_head(x)
        
        # Task 3: Path scoring (if path indices provided)
        path_scores = None
        if hasattr(data, 'path_indices') and data.path_indices is not None:
            path_scores = self._score_paths(x, data.path_indices)
        
        return {
            'edge_logits': edge_logits,
            'node_logits': node_logits,
            'path_scores': path_scores,
            'node_embeddings': x
        }
    
    def _score_paths(self, node_embeddings: torch.Tensor, 
                    path_indices: torch.Tensor) -> torch.Tensor:
        """
        Score paths using node embeddings.
        
        Args:
            node_embeddings: Node embeddings (N, hidden_dim)
            path_indices: Path node sequences (P, max_len)
        
        Returns:
            Path scores (P, 1)
        """
        scores = []
        for path in path_indices:
            if len(path) >= 3:
                # Use first, middle, and last node embeddings
                idx1 = path[0]
                idx2 = path[len(path) // 2]
                idx3 = path[-1]
                path_feat = torch.cat([
                    node_embeddings[idx1],
                    node_embeddings[idx2],
                    node_embeddings[idx3]
                ])
                score = self.path_head(path_feat)
                scores.append(score)
        
        if scores:
            return torch.stack(scores)
        return None
    
    def get_edge_probabilities(self, data: Data) -> Dict[int, float]:
        """
        Get edge probabilities from model output.
        
        Args:
            data: PyG Data object
        
        Returns:
            Dict mapping edge_id -> probability
        """
        with torch.no_grad():
            output = self.forward(data)
            edge_logits = output['edge_logits'].cpu().numpy().flatten()
        
        edge_probs = {}
        for i, prob in enumerate(edge_logits):
            edge_probs[i] = float(prob)
        
        return edge_probs
    
    def save(self, path: str):
        """Save model to disk."""
        state = {
            'model_state': self.state_dict(),
            'config': self.config
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PathGNNModel':
        """Load model from disk."""
        state = torch.load(path)
        config = state['config']
        model = cls(config)
        model.load_state_dict(state['model_state'])
        logger.info(f"Model loaded from {path}")
        return model


class SimpleGNN(PathGNNModel):
    """Lightweight GNN with 2 layers."""
    def __init__(self, config: Optional[GNNConfig] = None):
        if config is None:
            config = GNNConfig(num_layers=2, hidden_channels=32)
        else:
            config.num_layers = 2
            config.hidden_channels = 32
        super().__init__(config)


class MediumGNN(PathGNNModel):
    """Balanced GNN with 4 layers."""
    def __init__(self, config: Optional[GNNConfig] = None):
        if config is None:
            config = GNNConfig(num_layers=4, hidden_channels=64)
        else:
            config.num_layers = 4
            config.hidden_channels = 64
        super().__init__(config)


class DeepGNN(PathGNNModel):
    """Production GNN with 6 layers and residual connections."""
    def __init__(self, config: Optional[GNNConfig] = None):
        if config is None:
            config = GNNConfig(
                num_layers=6, 
                hidden_channels=128,
                use_residual=True,
                use_batch_norm=True
            )
        else:
            config.num_layers = 6
            config.hidden_channels = 128
            config.use_residual = True
            config.use_batch_norm = True
        super().__init__(config)


class GNNTrainer:
    """Trainer for GNN models."""
    
    def __init__(self, model: PathGNNModel, config: GNNConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss functions for each task
        self.edge_loss_fn = nn.BCELoss()
        self.node_loss_fn = nn.CrossEntropyLoss()
        self.path_loss_fn = nn.BCELoss()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Initialized GNNTrainer with {config}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_data in train_loader:
            batch_data = batch_data.to(self.device)
            
            output = self.model(batch_data)
            
            # Edge classification loss
            edge_logits = output['edge_logits']
            edge_labels = batch_data.edge_y if hasattr(batch_data, 'edge_y') else None
            if edge_labels is not None:
                edge_loss = self.edge_loss_fn(
                    edge_logits.squeeze(), 
                    edge_labels.float()
                )
            else:
                edge_loss = 0.0
            
            # Node classification loss
            node_logits = output['node_logits']
            node_labels = batch_data.node_y if hasattr(batch_data, 'node_y') else None
            if node_labels is not None:
                node_loss = self.node_loss_fn(node_logits, node_labels)
            else:
                node_loss = 0.0
            
            # Path scoring loss
            path_loss = 0.0
            if output['path_scores'] is not None and hasattr(batch_data, 'path_y'):
                path_loss = self.path_loss_fn(
                    output['path_scores'].squeeze(),
                    batch_data.path_y.float()
                )
            
            # Combined loss
            total_loss_batch = edge_loss + node_loss + path_loss
            
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Train loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(self.device)
                output = self.model(batch_data)
                
                # Compute validation loss
                edge_logits = output['edge_logits']
                edge_labels = batch_data.edge_y if hasattr(batch_data, 'edge_y') else None
                if edge_labels is not None:
                    edge_loss = self.edge_loss_fn(
                        edge_logits.squeeze(),
                        edge_labels.float()
                    )
                else:
                    edge_loss = 0.0
                
                total_loss += edge_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        logger.info(f"Val loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train model."""
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.config.epochs} - "
                           f"Train loss: {train_loss:.4f}")
