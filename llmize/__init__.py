"""
:no-index:
LLMize is a Python package that uses Large Language Models (LLMs) for multipurpose, numerical optimization tasks.
"""

__version__ = "0.2.0"

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