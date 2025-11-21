# -*- mode: python -*-
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Collect your non-Python files
datas = [
    ("setups.json", "."),
] + collect_data_files("images") + collect_data_files("bfconverter")

# Analysis: no need to include libpython manually
a = Analysis(
    ['main.py'],
    pathex=['.'],         # include current directory
    binaries=[('venv/lib/libpython3.13.so.1.0', '.')],          # leave empty; PyInstaller finds Python automatically
    datas=datas,
    hiddenimports=[],     # add dynamic imports if needed
    cipher=None,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    name="SFA_FECO_analyzer",
    debug=False,
    strip=False,
    upx=False,
    console=True,         # change to False if GUI
    onefile=True,         # bundle into a single executable
)