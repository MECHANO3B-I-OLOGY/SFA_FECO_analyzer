import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

class setup:
    def __init__(self, venv_dir="venv"):
        self.venv_dir = venv_dir

    def create_venv(self):
        """Create a virtual environment if it doesn't exist."""
        if not pathlib.Path(self.venv_dir).exists():
            print(f"Creating venv at {self.venv_dir}...")
            subprocess.run([sys.executable, "-m", "venv", self.venv_dir], check=True)
        else:
            print(f"Using existing venv at {self.venv_dir}")

    def get_venv_python(self):
        """Return path to the Python executable inside the venv."""
        if os.name == "nt":  # Windows
            return os.path.join(self.venv_dir, "Scripts", "python.exe")
        else:  # Linux/macOS
            return os.path.join(self.venv_dir, "bin", "python")

    def install_requirements(self, requirements_file="requirements.txt"):
        """Install dependencies inside the venv."""
        venv_python = self.get_venv_python()
        if pathlib.Path(requirements_file).exists():
            print(f"Installing requirements from {requirements_file}...")
            subprocess.run([venv_python, "-m", "pip", "install", "-r", requirements_file], check=True)
        else:
            print("⚠️ No requirements.txt found, skipping.")

    def drop_into_shell(self, venv_dir="venv"):
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

    def setup(self):
        self.create_venv()
        self.install_requirements()
        print("Setup complete!")

def load_internal():
    path = pathlib.Path("cache/internal.json")
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None
    return None

def save_internal(data):
    pathlib.Path("cache").mkdir(exist_ok=True)
    with open("cache/internal.json", "w") as f:
        json.dump(data, f, indent=2)

def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

class run:

    def __init__(self, venv_dir="venv"):
        self.venv_dir = venv_dir

    def get_venv_python(self):
        if os.name == "nt":
            return os.path.join(self.venv_dir, "Scripts", "python.exe")
        else:
            return os.path.join(self.venv_dir, "bin", "python")

    def runProgram(self):
        venv_python = self.get_venv_python()
        subprocess.run([venv_python, "main.py"])


if __name__ == "__main__":
    requirements_path = "requirements.txt"
    internal = load_internal()

    # Compute current requirements hash
    if pathlib.Path(requirements_path).exists():
        current_req_hash = hash_file(requirements_path)
    else:
        current_req_hash = None

    # CASE 1 — internal.json does NOT exist
    if internal is None:
        print("⚙️ No internal.json found — building fresh environment...")

        # Always recreate venv to ensure consistency
        if pathlib.Path("venv").exists():
            shutil.rmtree("venv")

        s = setup()
        s.setup()

        # save requirements hash to internal.json
        save_internal({"requirements": current_req_hash})

    # CASE 2 — internal.json exists
    else:
        stored_hash = internal.get("requirements", None)

        # requirements changed → rebuild venv
        if stored_hash != current_req_hash:
            print("🔁 Requirements changed — rebuilding venv...")

            if pathlib.Path("venv").exists():
                shutil.rmtree("venv")

            s = setup()
            s.setup()

            # update internal.json
            internal["requirements"] = current_req_hash
            save_internal(internal)

        else:
            print("✔ Requirements unchanged — using existing venv")

    # finally run program
    run = run()
    run.runProgram()