"""
Checkpoint management for StrandWeaver pipelines.

Handles creation, storage, and recovery of pipeline checkpoints.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging


class CheckpointManager:
    """
    Manage pipeline checkpoints for resumable execution.
    
    Features:
    - Create checkpoints after each step
    - Store intermediate files and metadata
    - List available checkpoints
    - Resume from specific checkpoint
    - Clean up old checkpoints
    """
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create(self, step_name: str, files: Dict[str, Path], 
               metadata: Optional[Dict[str, Any]] = None):
        """
        Create a checkpoint for a completed step.
        
        Args:
            step_name: Name of the completed step
            files: Dictionary of file paths to checkpoint
            metadata: Additional metadata to store
        """
        # Create checkpoint directory
        checkpoint_id = self._generate_checkpoint_id(step_name)
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files to checkpoint
        for file_key, file_path in files.items():
            if file_path and file_path.exists():
                dest = checkpoint_path / file_path.name
                shutil.copy2(file_path, dest)
                self.logger.debug(f"Checkpointed: {file_key} -> {dest}")
        
        # Save metadata
        checkpoint_metadata = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'files': {k: str(v) for k, v in files.items()},
            'metadata': metadata or {}
        }
        
        metadata_file = checkpoint_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        
        self.logger.info(f"Checkpoint created: {checkpoint_id}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = []
        
        for checkpoint_path in sorted(self.checkpoint_dir.glob("*")):
            if checkpoint_path.is_dir():
                metadata_file = checkpoint_path / 'metadata.json'
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    checkpoints.append({
                        'id': checkpoint_path.name,
                        'path': checkpoint_path,
                        **metadata
                    })
        
        return checkpoints
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint.
        
        Returns:
            Latest checkpoint metadata, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
    
    def restore(self, checkpoint_id: str, target_dir: Path):
        """
        Restore files from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            target_dir: Directory to restore files to
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        # Load metadata
        metadata_file = checkpoint_path / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Restore files
        for file_key, file_path in metadata['files'].items():
            source = checkpoint_path / Path(file_path).name
            if source.exists():
                dest = target_dir / Path(file_path).name
                shutil.copy2(source, dest)
                self.logger.debug(f"Restored: {file_key} -> {dest}")
        
        self.logger.info(f"Checkpoint restored: {checkpoint_id}")
        return metadata
    
    def remove(self, checkpoint_id: str):
        """
        Remove a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to remove
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            self.logger.info(f"Checkpoint removed: {checkpoint_id}")
    
    def clean_before(self, step_name: str):
        """
        Remove all checkpoints before a given step.
        
        Args:
            step_name: Remove checkpoints before this step
        """
        # TODO: Implement selective cleanup
        self.logger.info(f"Cleaning checkpoints before: {step_name}")
    
    def _generate_checkpoint_id(self, step_name: str) -> str:
        """
        Generate a unique checkpoint ID.
        
        Args:
            step_name: Name of the step
        
        Returns:
            Checkpoint ID (e.g., "001_profile")
        """
        # Count existing checkpoints
        existing = len(list(self.checkpoint_dir.glob("*")))
        checkpoint_num = existing + 1
        
        return f"{checkpoint_num:03d}_{step_name}"
