#!/usr/bin/env python3
"""
Test runner for braid library.

Run all tests with: python run_tests.py
Run specific test: python run_tests.py test_decompressor
"""

import sys
import unittest
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    # Discover and run tests
    if len(sys.argv) > 1:
        # Run specific test module
        test_name = sys.argv[1]
        if not test_name.startswith('test_'):
            test_name = f'test_{test_name}'

        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(f'tests.{test_name}')
    else:
        # Run all tests
        loader = unittest.TestLoader()
        suite = loader.discover('tests', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)