import os
import sys
import subprocess
import pathlib
from setup import setup

def get_venv_python(venv_dir="venv"):
    """Return path to the Python executable inside the venv."""
    if os.name == "nt":  # Windows
        return os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Linux/macOS
        return os.path.join(venv_dir, "bin", "python")

def runProgram(venv_dir="venv"):
    """Run the main script"""
    venv_python = get_venv_python(venv_dir)

    subprocess.run([venv_python, "main.py"])

if __name__ == "__main__":
    if not pathlib.Path("venv").exists():
        setup()
    else:
        print(f"Using existing venv at venv")

    runProgram("venv")
