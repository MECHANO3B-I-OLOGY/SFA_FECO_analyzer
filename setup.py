import os
import sys
import subprocess
import pathlib
import shutil

def create_venv(venv_dir="venv"):
    """Create a virtual environment if it doesn't exist."""
    if not pathlib.Path(venv_dir).exists():
        print(f"Creating venv at {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    else:
        print(f"Using existing venv at {venv_dir}")

def get_venv_python(venv_dir="venv"):
    """Return path to the Python executable inside the venv."""
    if os.name == "nt":  # Windows
        return os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Linux/macOS
        return os.path.join(venv_dir, "bin", "python")

def install_requirements(venv_dir="venv", requirements_file="requirements.txt"):
    """Install dependencies inside the venv."""
    venv_python = get_venv_python(venv_dir)
    if pathlib.Path(requirements_file).exists():
        print(f"Installing requirements from {requirements_file}...")
        subprocess.run([venv_python, "-m", "pip", "install", "-r", requirements_file], check=True)
    else:
        print("⚠️ No requirements.txt found, skipping.")

def drop_into_shell(venv_dir="venv"):
    """Drop the user into a shell with the venv activated."""
    if os.name == "nt":  # Windows
        # Detect if running in PowerShell or cmd
        parent_proc = os.environ.get("COMSPEC", "").lower()
        powershell_exe = shutil.which("powershell.exe")

        if "powershell" in parent_proc and powershell_exe:
            activate_script = os.path.join(venv_dir, "Scripts", "Activate.ps1")
            print("➡️ Dropping into PowerShell with venv activated...")
            subprocess.run([powershell_exe, "-NoExit", "-ExecutionPolicy", "Bypass", "-File", activate_script])
        else:
            activate_script = os.path.join(venv_dir, "Scripts", "activate.bat")
            print("➡️ Dropping into cmd.exe with venv activated...")
            subprocess.run(["cmd.exe", "/K", activate_script])
    else:  # Linux/macOS
        activate_script = os.path.join(venv_dir, "bin", "activate")
        print("➡️ Dropping into bash with venv activated...")
        subprocess.run(["bash", "--rcfile", activate_script, "-i"])  # interactive shell

if __name__ == "__main__":
    venv_dir = "venv"
    create_venv(venv_dir)
    install_requirements(venv_dir)
    print("Setup complete! Launching shell...")
    drop_into_shell(venv_dir)