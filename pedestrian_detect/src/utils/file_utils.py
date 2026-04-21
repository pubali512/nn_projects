"""File and directory utility functions."""

import os
import glob
import shutil
from pathlib import Path


def list_images(directory: str, pattern: str = "*.pgm") -> list:
    """List image files in a directory matching a pattern.

    Args:
        directory: Path to the directory.
        pattern: Glob pattern (e.g. '*.pgm').

    Returns:
        Sorted list of absolute file paths.
    """
    patterns = [p.strip() for p in pattern.split(",")]
    files = set()
    for p in patterns:
        matched = glob.glob(os.path.join(directory, p))
        files.update(matched)
    return sorted(files)


def dir_contains_images(directory: str) -> bool:
    """Check if a directory exists and contains at least one file."""
    if not os.path.isdir(directory):
        return False
    try:
        entries = os.listdir(directory)
        return len(entries) > 0
    except OSError:
        return False


def ensure_dir(path: str) -> bool:
    """Create directory if it doesn't exist.

    Returns:
        True if directory exists or was created successfully.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError:
        return False


def clean_dir(path: str) -> bool:
    """Delete all files in a directory (not subdirectories).

    If the directory doesn't exist, create it.

    Returns:
        True on success.
    """
    try:
        if os.path.isdir(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        else:
            os.makedirs(path)
        return True
    except OSError:
        return False


def create_or_clean_dir(path: str) -> bool:
    """Create directory or clean existing one.

    Returns:
        True on success.
    """
    return clean_dir(path)


def read_file_list(filepath: str) -> list:
    """Read a list of file paths from a text file (one per line).

    Args:
        filepath: Path to the text file.

    Returns:
        List of valid file paths.
    """
    files = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and os.path.isfile(line):
                if line not in files:
                    files.append(line)
    return files
