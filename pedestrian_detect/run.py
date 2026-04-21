#!/usr/bin/env python3
"""
run.py — Cross-platform launcher for the Pedestrian Detect app.

Usage:
    python run.py           # Launch the app using the .venv Python
    python run.py --setup   # Run setup first, then launch
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_venv_python(project_root: Path) -> Path:
    venv_dir = project_root / ".venv"
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main():
    project_root = get_project_root()
    venv_python = get_venv_python(project_root)
    src_dir = project_root / "src"
    main_script = src_dir / "main.py"

    # Optionally run setup first
    if "--setup" in sys.argv:
        setup_script = project_root / "setup_env.py"
        if setup_script.is_file():
            print("Running setup_env.py first ...\n")
            result = subprocess.run([sys.executable, str(setup_script)])
            if result.returncode != 0:
                print("\nSetup failed. Aborting.")
                sys.exit(1)
            print()
        else:
            print("WARNING: setup_env.py not found. Skipping setup.\n")

    # Check venv exists
    if not venv_python.is_file():
        print(f"ERROR: Virtual environment not found at .venv/")
        print(f"       Run 'python setup_env.py' first to create it.")
        sys.exit(1)

    # Check main.py exists
    if not main_script.is_file():
        print(f"ERROR: src/main.py not found.")
        sys.exit(1)

    # Launch the app from src/ directory using the venv Python
    print(f"Launching Pedestrian Detect ...")
    print(f"  Python: {venv_python}")
    print(f"  App:    {main_script}\n")

    result = subprocess.run(
        [str(venv_python), str(main_script)],
        cwd=str(src_dir),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
