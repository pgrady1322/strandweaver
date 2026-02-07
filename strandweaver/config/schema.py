"""
StrandWeaver v0.1.0

Configuration schema for StrandWeaver.

Defines all available configuration parameters with defaults and validation.

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml


# Default configuration values
DEFAULT_CONFIG = {
    # ========================================================================
    # AI/ML Settings (DEFAULT: ENABLED)
    # ========================================================================
    'ai': {
        'enabled': True,  # Master AI switch (use ML models by default)
        'use_classical_fallback': True,  # Fall back to heuristics if model fails
        
        # Error Correction AI
        'correction': {
            'adaptive_kmer': {
                'enabled': True,
                'model_path': None,  # Auto-detect from installation
                'min_confidence': 0.7,
            },
            'base_error_classifier': {
                'enabled': True,
                'model_path': None,
                'min_confidence': 0.8,
            },
        },
        
        # Assembly AI Models
        'assembly': {
            'edge_ai': {
                'enabled': True,
                'model_path': None,
                'edge_threshold': 0.5,
            },
            'path_gnn': {
                'enabled': True,
                'model_path': None,
                'min_path_score': 0.6,
            },
            'diploid_ai': {
                'enabled': True,
                'model_path': None,
                'phase_confidence': 0.7,
            },
            'ul_routing_ai': {
                'enabled': True,
                'model_path': None,
                'anchor_threshold': 0.6,
            },
            'sv_ai': {
                'enabled': True,
                'model_path': None,
                'sv_confidence': 0.75,
            },
        },
        
        # Claude API (experimental)
        'claude': {
            'enabled': False,
            'api_key': None,  # Set via environment or config
            'model': 'claude-3-opus-20240229',
            'use_for_finishing': False,
        },
    },
    
    # ========================================================================
    # Hardware Settings (DEFAULT: CPU)
    # ========================================================================
    'hardware': {
        'use_gpu': False,  # CPU by default
        'gpu_device': 0,  # CUDA device if GPU enabled
        'threads': None,  # Auto-detect from system
        'memory_limit_gb': None,  # No limit by default
    },
    
    # ========================================================================
    # Pipeline Control
    # ========================================================================
    'pipeline': {
        'steps': ['kweaver', 'profile', 'correct', 'assemble', 'finish', 'misassembly_report', 'classify_chromosomes'],
        'skip_profiling': False,
        'skip_correction': False,
        'resume': False,
        'checkpoint_interval': 'per_step',  # 'per_step', 'per_contig', 'disabled'
        'checkpoint_dir': None,  # Default: <output>/checkpoints
    },
    
    # ========================================================================
    # Error Profiling
    # ========================================================================
    'profiling': {
        'sample_size': 100000,  # Number of reads to sample
        'min_quality': 0,
        'technology_detection': {
            'enabled': True,
            'confidence_threshold': 0.8,
        },
        'ancient_dna': {
            'detect_damage': True,
            'ct_threshold': 0.05,  # C->T frequency for damage detection
            'ga_threshold': 0.05,  # G->A frequency
        },
    },
    
    # ========================================================================
    # Error Correction
    # ========================================================================
    'correction': {
        # Technology-specific settings
        'illumina': {
            'kmer_size': 31,
            'min_kmer_count': 2,
            'error_rate': 0.01,
            'use_ai_kmer_selection': True,  # Use AdaptiveKmerAI
        },
        'ancient_dna': {
            'damage_correction': True,
            'deamination_mode': 'conservative',  # 'conservative', 'aggressive', 'ai'
            'use_ai_damage_classifier': True,  # Use BaseErrorClassifierAI
            'trim_ends': True,
            'trim_length': 5,
        },
        'ont': {
            'error_rate': 0.10,  # ONT R9/R10 default
            'homopolymer_correction': True,
            'use_ai_error_model': True,
        },
        'pacbio': {
            'error_rate': 0.01,  # HiFi default
            'use_ai_error_model': True,
        },
        
        # Multi-pass correction
        'max_iterations': 3,
        'convergence_threshold': 0.001,
    },
    
    # ========================================================================
    # Assembly
    # ========================================================================
    'assembly': {
        # Graph construction
        'graph': {
            'type': 'auto',  # 'dbg', 'string', 'auto'
            'kmer_size': None,  # Auto-select with AI if None
            'min_overlap': 50,
            'min_coverage': 2,
            'use_edge_ai': True,  # AI-powered edge filtering
        },
        
        # Overlap Layout Consensus (for short reads)
        'olc': {
            'enabled': 'auto',  # True, False, 'auto'
            'min_overlap_length': 40,
            'min_identity': 0.95,
            'max_overhang': 10,
        },
        
        # De Bruijn Graph
        'dbg': {
            'enabled': True,
            'adaptive_k': True,  # Use AdaptiveKmerAI for k selection
            'k_range': [21, 127],  # Min and max k values
            'k_step': 10,
            'coverage_cutoff': 'auto',
            'use_path_gnn': True,  # AI-powered path resolution
        },
        
        # String Graph (for ultra-long reads)
        'string_graph': {
            'enabled': 'auto',  # Auto-enable if UL reads present
            'min_read_length': 50000,
            'use_ul_routing_ai': True,  # AI-powered UL read routing
        },
        
        # Diploid/Polyploid Assembly
        'diploid': {
            'mode': 'auto',  # 'haploid', 'diploid', 'polyploid', 'auto'
            'use_diploid_ai': True,  # AI-powered phasing
            'min_allele_frequency': 0.25,
        },
        
        # Structural Variant Detection
        'sv_detection': {
            'enabled': True,
            'use_sv_ai': True,  # AI-powered SV detection
            'min_sv_size': 50,
            'max_sv_size': 1000000,
        },
        
        # Contig building
        'contigs': {
            'min_length': 500,
            'min_coverage': 2,
            'quality_threshold': 20,
        },
    },
    
    # ========================================================================
    # Scaffolding
    # ========================================================================
    'scaffolding': {
        'enabled': 'auto',  # Auto-enable if Hi-C data present
        
        # Hi-C scaffolding
        'hic': {
            'enabled': False,
            'min_mapq': 30,
            'contact_threshold': 5,
            'resolution': 10000,  # Base pairs
        },
        
        # Ultra-long read scaffolding
        'ultralong': {
            'enabled': 'auto',
            'min_read_length': 50000,
            'min_anchors': 2,
        },
    },
    
    # ========================================================================
    # Finishing
    # ========================================================================
    'finishing': {
        'enabled': True,
        
        # Gap filling
        'gap_filling': {
            'enabled': True,
            'max_gap_size': 10000,
            'min_overlap_to_gap': 100,
        },
        
        # Polishing
        'polishing': {
            'enabled': True,
            'rounds': 2,
            'use_original_reads': True,
        },
        
        # Decontamination
        'decontamination': {
            'enabled': False,
            'reference_databases': [],  # Paths to contamination DBs
        },
    },
    
    # ========================================================================
    # Output
    # ========================================================================
    'output': {
        'format': 'fasta',  # 'fasta', 'fastq', 'both'
        'compression': 'gzip',  # 'none', 'gzip', 'bzip2'
        'include_quality': True,
        'include_metadata': True,
        
        # Visualization
        'visualization': {
            'generate_bandage_files': True,
            'graph_format': 'gfa',  # 'gfa', 'fastg'
        },
        
        # Logging
        'logging': {
            'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
            'log_file': 'strandweaver.log',
        },
    },
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from file or return defaults.
    
    Args:
        config_path: Path to YAML config file (None = use defaults)
    
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
            
            # Deep merge user config into defaults
            config = _deep_merge(config, user_config)
    
    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
    
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config_template(output_path: Path, template: str = 'default'):
    """
    Save a configuration template to file.
    
    Args:
        output_path: Output file path
        template: Template type ('default', 'illumina', 'ancient', 'ont', 'pacbio', 'hybrid')
    """
    config = DEFAULT_CONFIG.copy()
    
    # Customize for specific templates
    if template == 'illumina':
        config['pipeline']['steps'] = ['kweaver', 'profile', 'correct', 'assemble', 'finish', 'misassembly_report', 'classify_chromosomes']
        config['assembly']['graph']['type'] = 'dbg'
        config['assembly']['olc']['enabled'] = True
        config['correction']['illumina']['use_ai_kmer_selection'] = True
        
    elif template == 'ancient':
        config['profiling']['ancient_dna']['detect_damage'] = True
        config['correction']['ancient_dna']['damage_correction'] = True
        config['correction']['ancient_dna']['use_ai_damage_classifier'] = True
        
    elif template == 'ont':
        config['assembly']['graph']['type'] = 'dbg'
        config['correction']['ont']['use_ai_error_model'] = True
        config['assembly']['string_graph']['enabled'] = True
        
    elif template == 'pacbio':
        config['assembly']['graph']['type'] = 'dbg'
        config['correction']['pacbio']['use_ai_error_model'] = True
        
    elif template == 'hybrid':
        config['assembly']['graph']['type'] = 'auto'
        config['assembly']['diploid']['mode'] = 'auto'
        config['scaffolding']['enabled'] = True
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate AI settings
    if config.get('ai', {}).get('enabled') and config.get('hardware', {}).get('use_gpu'):
        # Check GPU availability if requested
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if not has_cuda and not has_mps:
                errors.append("GPU requested but neither CUDA nor MPS (Apple Silicon) available")
        except ImportError:
            errors.append("GPU requested but PyTorch not installed")
    
    # Validate pipeline steps
    valid_steps = ['kweaver', 'profile', 'correct', 'assemble', 'finish', 'misassembly_report', 'classify_chromosomes']
    for step in config.get('pipeline', {}).get('steps', []):
        if step not in valid_steps:
            errors.append(f"Invalid pipeline step: {step}")
    
    # Validate k-mer ranges
    k_range = config.get('assembly', {}).get('dbg', {}).get('k_range', [21, 127])
    if len(k_range) != 2 or k_range[0] >= k_range[1]:
        errors.append(f"Invalid k_range: must be [min, max] with min < max")
    
    # Validate model paths if AI enabled
    if config.get('ai', {}).get('enabled'):
        for model_type in ['correction', 'assembly']:
            model_configs = config.get('ai', {}).get(model_type, {})
            for model_name, model_settings in model_configs.items():
                if isinstance(model_settings, dict) and model_settings.get('enabled'):
                    model_path = model_settings.get('model_path')
                    if model_path and not Path(model_path).exists():
                        errors.append(f"Model path not found: {model_path} ({model_type}.{model_name})")
    
    return errors
