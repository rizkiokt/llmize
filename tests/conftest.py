"""
pytest configuration file
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# This won't override existing environment variables (like those set in CI)
load_dotenv()
