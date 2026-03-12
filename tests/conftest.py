"""Pytest global test environment configuration."""

import os
import sys
from pathlib import Path
import tempfile

# Ensure Numba caching does not require repository write access during tests
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())
# Prefer pure-Python execution in tests to avoid cache locator issues
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Add parent directory to path for imports
dd_startup_root = Path(__file__).parent.parent
sys.path.insert(0, str(dd_startup_root))
