#!/usr/bin/env python3
"""
setup_env.py — Cross-platform project setup for Pedestrian Detect

Works on Windows, macOS, and Linux.

Usage:
    python setup_env.py          # Create venv and install dependencies
    python setup_env.py --clean  # Remove existing venv and start fresh
"""

import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
VENV_DIR = ".venv"
REQUIREMENTS = os.path.join("src", "requirements.txt")
# ───────────────────────────────────────────────────────────────


def get_project_root() -> Path:
    """Return the directory where this script lives (project root)."""
    return Path(__file__).resolve().parent


def get_venv_python(venv_path: Path) -> Path:
    """Return the path to the Python executable inside the venv."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def get_venv_pip(venv_path: Path) -> Path:
    """Return the path to the pip executable inside the venv."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    return venv_path / "bin" / "pip"


def get_activate_command(venv_path: Path) -> str:
    """Return the shell command to activate the venv."""
    if platform.system() == "Windows":
        return f"  {venv_path}\\Scripts\\activate"
    return f"  source {venv_path}/bin/activate"


def print_header(msg: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}")


def print_step(msg: str) -> None:
    print(f"  -> {msg}")


def run_command(cmd: list, description: str) -> None:
    """Run a subprocess command, raising on failure."""
    print_step(description)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ERROR: {description} failed!")
        if result.stderr:
            print(f"  STDERR: {result.stderr.strip()}")
        sys.exit(1)


def main():
    # Parse args
    clean = "--clean" in sys.argv

    project_root = get_project_root()
    os.chdir(project_root)

    venv_path = project_root / VENV_DIR
    requirements_path = project_root / REQUIREMENTS

    print_header("Pedestrian Detect — Project Setup")

    # System info
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.version.split()[0]} ({sys.executable})")
    print(f"  Project  : {project_root}")

    # Verify requirements file exists
    if not requirements_path.is_file():
        print(f"\n  ERROR: Requirements file not found: {requirements_path}")
        sys.exit(1)

    # Clean if requested
    if clean and venv_path.exists():
        print_step(f"Removing existing venv at {venv_path} ...")
        shutil.rmtree(venv_path)
        print_step("Removed.")

    # Create virtual environment
    if venv_path.exists():
        print_step(f"Virtual environment already exists at {VENV_DIR}/ — skipping creation.")
    else:
        print_step(f"Creating virtual environment in {VENV_DIR}/ ...")
        venv.create(str(venv_path), with_pip=True)
        print_step("Virtual environment created.")

    venv_python = get_venv_python(venv_path)
    venv_pip = get_venv_pip(venv_path)

    # Verify venv Python works
    if not venv_python.is_file():
        print(f"\n  ERROR: venv Python not found at {venv_python}")
        sys.exit(1)

    # Upgrade pip
    run_command(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        "Upgrading pip ...",
    )

    # Install dependencies
    run_command(
        [str(venv_pip), "install", "-r", str(requirements_path), "--quiet"],
        f"Installing dependencies from {REQUIREMENTS} ...",
    )

    # Verify key imports
    print_step("Verifying installed packages ...")
    verify = subprocess.run(
        [str(venv_python), "-c", "import cv2, numpy, sklearn, PIL; print('OK')"],
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0 or "OK" not in verify.stdout:
        print("  WARNING: Package verification failed. Some imports may not work.")
        if verify.stderr:
            print(f"  {verify.stderr.strip()}")
    else:
        print_step("All packages verified successfully.")

    # Done
    print_header("Setup Complete!")
    print()
    print("  To activate the virtual environment, run:")
    print()
    print(get_activate_command(venv_path))
    print()
    print("  Then launch the app:")
    print()
    if platform.system() == "Windows":
        print("  cd src")
        print("  python main.py")
    else:
        print("  cd src && python main.py")
    print()


if __name__ == "__main__":
    main()
