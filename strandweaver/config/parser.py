#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.1.0

Configuration parser — YAML config loading, merging, and validation.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigParser:
    """
    Parse and validate StrandWeaver configuration files.
    
    Features:
    - Load YAML configuration files
    - Merge with default values
    - Environment variable substitution (${VAR} and ${VAR:-default})
    - CLI parameter overrides
    - Schema validation
    - Dotted notation access (e.g., config.get('assembly.min_overlap'))
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration parser.
        
        Args:
            config_file: Path to YAML configuration file (optional)
        """
        self.config_file = Path(config_file) if config_file else None
        self._config: Dict[str, Any] = {}
        
        # Load default configuration
        self._load_defaults()
        
        # Load user configuration if provided
        if self.config_file:
            self._load_user_config()
    
    def _load_defaults(self):
        """Load default configuration values."""
        defaults_path = Path(__file__).parent / "defaults.yaml"
        
        if defaults_path.exists():
            with open(defaults_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Fallback minimal defaults
            self._config = {
                'input': {},
                'profiling': {},
                'correction': {},
                'assembly': {},
                'ai_finishing': {},
                'output': {},
                'execution': {}
            }
    
    def _load_user_config(self):
        """Load and merge user configuration file."""
        if not self.config_file or not self.config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_file}"
            )
        
        try:
            with open(self.config_file, 'r') as f:
                user_config = yaml.safe_load(f)
            
            if user_config:
                # Merge user config with defaults (user values override defaults)
                self._config = self._deep_merge(self._config, user_config)
                
                # Substitute environment variables
                self._config = self._substitute_env_vars(self._config)
        
        except yaml.YAMLError as e:
            raise ConfigValidationError(
                f"Invalid YAML in config file {self.config_file}: {e}"
            )
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary (defaults)
            override: Override dictionary (user values)
        
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Supports:
        - ${VAR}: Replace with environment variable VAR
        - ${VAR:-default}: Replace with VAR, or 'default' if not set
        
        Args:
            config: Configuration value (can be dict, list, or string)
        
        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        
        elif isinstance(config, str):
            # Pattern: ${VAR} or ${VAR:-default}
            pattern = r'\$\{([^}:]+)(?::-(.*?))?\}'
            
            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2)
                
                # Get environment variable, or use default
                return os.environ.get(var_name, default_value or '')
            
            return re.sub(pattern, replace_var, config)
        
        else:
            return config
    
    def merge_cli_overrides(self, overrides: Dict[str, Any]):
        """
        Merge command-line overrides into configuration.
        
        Args:
            overrides: Dictionary of override values
                      Keys can use dotted notation (e.g., 'assembly.min_overlap')
        """
        for key, value in overrides.items():
            # Handle dotted notation (e.g., 'assembly.min_overlap')
            keys = key.split('.')
            
            # Navigate to the nested dictionary
            target = self._config
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            # Set the value
            target[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports dotted notation for nested access.
        
        Args:
            key: Configuration key (e.g., 'assembly.min_overlap')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_input_config(self) -> Dict[str, Any]:
        """Get input configuration section."""
        return self._config.get('input', {})
    
    def get_profiling_config(self) -> Dict[str, Any]:
        """Get profiling configuration section."""
        return self._config.get('profiling', {})
    
    def get_correction_config(self) -> Dict[str, Any]:
        """Get correction configuration section."""
        return self._config.get('correction', {})
    
    def get_assembly_config(self) -> Dict[str, Any]:
        """Get assembly configuration section."""
        return self._config.get('assembly', {})
    
    def get_ai_finishing_config(self) -> Dict[str, Any]:
        """Get AI finishing configuration section."""
        return self._config.get('ai_finishing', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration section."""
        return self._config.get('output', {})
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration section."""
        return self._config.get('execution', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    def validate(self, schema_file: Optional[Union[str, Path]] = None) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            schema_file: Path to schema YAML file (optional)
        
        Returns:
            True if valid
        
        Raises:
            ConfigValidationError: If validation fails
        """
        # TODO: Implement schema validation
        # For now, just check that required sections exist
        required_sections = ['input', 'output', 'execution']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigValidationError(
                    f"Missing required configuration section: {section}"
                )
        
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigParser(config_file={self.config_file})"

# StrandWeaver v0.1.0
# Any usage is subject to this software's license.
