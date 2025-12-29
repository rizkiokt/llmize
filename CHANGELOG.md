# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-29

### Added
- **Configuration System**: Introduced a centralized configuration management system
  - Added `llmize.config` module with `Config` class
  - Support for TOML configuration files (`llmize.toml`)
  - Environment variable overrides for all configuration options
  - Default configuration file with sensible defaults
  - Configuration loading from multiple locations (user home, system-wide, project)

- **Default Model Management**: 
  - Changed default LLM model from `gemini-2.5-flash-lite` to `gemma-3-27b-it`
  - All hardcoded defaults replaced with configuration-driven defaults
  - Support for both `gemini` and `gemma` model prefixes in client initialization

- **Enhanced Error Handling**:
  - Improved error messages for invalid optimization types
  - Better handling of hyperparameter parsing edge cases
  - More resilient parsing of LLM responses

- **Documentation**:
  - Added configuration documentation
  - Updated usage examples to reflect new defaults

### Changed
- **API Changes**:
  - All optimizer methods now accept `None` for optional parameters and use config defaults
  - `llm_call.generate_content()` uses config defaults for temperature, retries, and retry delay
  - Base class methods (`maximize`, `minimize`, `get_sample_prompt`) use config defaults
  - All optimizer classes (OPRO, ADOPRO, HLMEA, HLMSA) use config defaults

- **Dependencies**:
  - Added `toml>=0.10.2` requirement for configuration file support
  - Updated `google-genai` requirement to `>=1.15.0`

### Fixed
- Fixed hyperparameter parsing to handle different closing tag formats (`</hp>`, `<\hp>`, `<\\hp>`)
- Fixed error message consistency across all optimizer methods
- Fixed test assertions to match new configuration-driven defaults
- Improved retry logic in solution generation to be more resilient

### Deprecated
- Hardcoded default values in method signatures (replaced with config-driven defaults)

## [0.1.5] - 2024-12-28

### Added
- Initial implementation of HLMSA (Hyper-heuristic LLM-driven Simulated Annealing)
- Support for hyperparameter adaptation in optimization
- Improved test coverage for all optimizers

### Fixed
- Bug fixes for parallel evaluation
- Improved parsing of complex solution formats

## [0.1.4] - 2024-12-27

### Added
- ADOPRO (Adaptive OPRO) optimizer implementation
- Support for adaptive temperature and batch size
- Enhanced callback system

### Changed
- Improved optimization performance
- Better logging and debugging support

## [0.1.3] - 2024-12-26

### Added
- HLMEA (Hyper-heuristic LLM-driven Evolutionary Algorithm) optimizer
- Support for evolutionary operations (crossover, mutation)
- Population-based optimization

### Fixed
- Fixed initialization issues with multiple optimizers
- Improved solution parsing for complex formats

## [0.1.2] - 2024-12-25

### Added
- Initial public release
- OPRO (Optimization by PROmpting) optimizer
- Basic optimization framework
- Support for maximize/minimize operations
- Parallel evaluation capabilities
- Comprehensive test suite
