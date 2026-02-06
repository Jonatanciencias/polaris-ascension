#!/usr/bin/env python3
"""
License Header Checker for Radeon RX 580 Energy-Efficient Computing Framework

This script checks that all Python files have proper license headers.
"""

import os
import sys
from pathlib import Path

# Expected license header
LICENSE_HEADER = '''# Copyright (c) 2024 Jonathan Ciencias
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''

def check_file_has_license_header(file_path):
    """Check if a file has the proper license header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if the license header is present
        return LICENSE_HEADER.strip() in content
    except (UnicodeDecodeError, IOError):
        return False

def find_python_files(directory):
    """Find all Python files in the directory tree."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist', 'venv', 'env', '.venv', '.tox']]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files

def main():
    """Main function."""
    # Get the project root directory
    project_root = Path(__file__).parent

    # Find all Python files
    python_files = find_python_files(project_root)

    # Filter out files that shouldn't have license headers
    exclude_patterns = [
        'setup.py',
        'conftest.py',
        'docs/paper/',
        'build/',
        'dist/',
        'src/_version.py',
    ]

    files_to_check = []
    for file_path in python_files:
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern in file_path:
                should_exclude = True
                break
        if not should_exclude:
            files_to_check.append(file_path)

    # Check each file
    missing_headers = []
    for file_path in files_to_check:
        if not check_file_has_license_header(file_path):
            missing_headers.append(file_path)

    # Report results
    if missing_headers:
        print("❌ The following files are missing license headers:")
        for file_path in missing_headers:
            print(f"  - {file_path}")
        print(f"\nExpected license header:\n{LICENSE_HEADER}")
        return 1
    else:
        print("✅ All Python files have proper license headers!")
        return 0

if __name__ == '__main__':
    sys.exit(main())