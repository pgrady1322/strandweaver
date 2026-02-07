"""
StrandWeaver v0.1.0

User-Configurable Training Infrastructure

This module provides a flexible, parameter-driven training data generation system
for StrandWeaver ML models. Unlike the scenario-based training system, this allows
users to specify exact parameters for:

- Genome characteristics (size, GC content, repeat density)
- Population diversity (number of genomes, ploidy)
- Sequencing technologies and coverage
- Structural variant density and types

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from .training_config import (
    UserGenomeConfig,
    UserReadConfig,
    UserTrainingConfig,
    ReadType
)

from .config_based_workflow import (
    generate_custom_training_data,
    TrainingDataGenerator
)

__all__ = [
    'UserGenomeConfig',
    'UserReadConfig',
    'UserTrainingConfig',
    'ReadType',
    'generate_custom_training_data',
    'TrainingDataGenerator'
]
