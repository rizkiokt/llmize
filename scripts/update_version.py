import re
import sys
from pathlib import Path

def update_version(new_version):
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        content = re.sub(
            r'version = "[0-9]+\.[0-9]+\.[0-9]+"',
            f'version = "{new_version}"',
            content
        )
        pyproject_path.write_text(content)

    # Update setup.py
    setup_path = Path("setup.py")
    if setup_path.exists():
        content = setup_path.read_text()
        content = re.sub(
            r'version="[0-9]+\.[0-9]+\.[0-9]+"',
            f'version="{new_version}"',
            content
        )
        setup_path.write_text(content)

    # Update __init__.py if it exists
    init_path = Path("llmize/__init__.py")
    if init_path.exists():
        content = init_path.read_text()
        content = re.sub(
            r'__version__ = "[0-9]+\.[0-9]+\.[0-9]+"',
            f'__version__ = "{new_version}"',
            content
        )
        init_path.write_text(content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <new_version>")
        sys.exit(1)
    
    new_version = sys.argv[1]
    update_version(new_version) 