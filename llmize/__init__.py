"""
LLMize - A library to use Large Language Models (LLMs) as numerical optimizers
"""

__version__ = "0.1.2"

from .methods.opro import OPRO
from .methods.adopro import ADOPRO
from .methods.hlmea import HLMEA
from .methods.hlmsa import HLMSA

# Expose modules
from . import methods

# Expose main classes
__all__ = [
    'OPRO',
    'ADOPRO',
    'HLMEA',
    'HLMSA',
    'methods'
]