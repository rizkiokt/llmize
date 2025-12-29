"""
Configuration management for llmize.
"""

import os
from typing import Optional
from pathlib import Path
import toml


class Config:
    """Configuration class for llmize settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to config file. If None, looks for default locations.
        """
        self.config = {}
        self._load_config(config_file)
    
    def _load_config(self, config_file: Optional[str] = None):
        """Load configuration from file."""
        # Default config file locations to check
        config_paths = [
            config_file,
            os.path.expanduser("~/.llmize/config.toml"),
            "/etc/llmize/config.toml",
            "llmize.toml",
            ".llmize.toml",
        ]
        
        # Filter out None values
        config_paths = [p for p in config_paths if p is not None]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    self.config = toml.load(path)
                    break
                except Exception as e:
                    print(f"Warning: Failed to load config from {path}: {e}")
        
        # Set defaults if not in config
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            "llm": {
                "default_model": os.getenv("LLMIZE_DEFAULT_MODEL", "gemma-3-27b-it"),
                "temperature": float(os.getenv("LLMIZE_TEMPERATURE", 1.0)),
                "max_retries": int(os.getenv("LLMIZE_MAX_RETRIES", 10)),
                "retry_delay": int(os.getenv("LLMIZE_RETRY_DELAY", 5)),
            },
            "optimization": {
                "default_num_steps": 50,
                "default_batch_size": 5,
                "parallel_n_jobs": 1,
            }
        }
        
        # Merge defaults with config (config takes precedence)
        for section, values in defaults.items():
            if section not in self.config:
                self.config[section] = {}
            for key, value in values.items():
                if key not in self.config[section]:
                    self.config[section][key] = value
    
    def get(self, key: str, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'llm.default_model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def default_model(self) -> str:
        """Get the default LLM model."""
        return self.get("llm.default_model", "gemma-3-27b-it")
    
    @property
    def temperature(self) -> float:
        """Get the default temperature."""
        return self.get("llm.temperature", 1.0)
    
    @property
    def max_retries(self) -> int:
        """Get the default max retries."""
        return self.get("llm.max_retries", 10)
    
    @property
    def retry_delay(self) -> int:
        """Get the default retry delay."""
        return self.get("llm.retry_delay", 5)
    
    @property
    def default_num_steps(self) -> int:
        """Get the default number of optimization steps."""
        return self.get("optimization.default_num_steps", 50)
    
    @property
    def default_batch_size(self) -> int:
        """Get the default batch size."""
        return self.get("optimization.default_batch_size", 5)
    
    @property
    def parallel_n_jobs(self) -> int:
        """Get the default number of parallel jobs."""
        return self.get("optimization.parallel_n_jobs", 1)


# Global config instance
_config = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_file: Path to config file (only used on first call)
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


def reload_config(config_file: Optional[str] = None):
    """
    Reload the configuration.
    
    Args:
        config_file: Path to config file
    """
    global _config
    _config = Config(config_file)
