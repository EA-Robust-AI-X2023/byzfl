import os
import glob

from loguru import logger
import pytest

def find_test_files():
    paths = ['.', './tests']
    test_files = []
    for path in paths:
        pattern = os.path.join(path, 'unit_test_*.py')
        test_files.extend(glob.glob(pattern))
    return test_files

if __name__ == '__main__':
    test_files = find_test_files()
    if not test_files:
        logger.critical("No unit test files found.")
    else:
        logger.info(f"Running tests from: {test_files}")
        pytest.main(test_files)