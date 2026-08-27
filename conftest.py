"""Make the repository root importable so `import BO_ML4F` works from tests/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
