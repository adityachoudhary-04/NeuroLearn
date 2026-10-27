import zipfile
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ZIP = ROOT_DIR / "neurolearn_rag_project.zip"
DESKTOP_ZIP = ROOT_DIR.parent / "neurolearn_rag_project.zip"

EXCLUDE_DIRS = {"venv", ".venv", "__pycache__", ".pytest_cache", ".git", "test_chroma"}
EXCLUDE_EXTENSIONS = {".pyc", ".pyo", ".pyd", ".zip"}

def make_zip(zip_target: Path):
    print(f"Creating ZIP archive: {zip_target}")
    with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in ROOT_DIR.rglob("*"):
            # Skip target zip itself
            if file_path.suffix in EXCLUDE_EXTENSIONS:
                continue
            
            # Check if any parent directory is excluded
            parts = file_path.relative_to(ROOT_DIR).parts
            if any(part in EXCLUDE_DIRS for part in parts):
                continue
            
            # Skip generated database binary files in data/chroma_db except .gitkeep
            if len(parts) > 2 and parts[0] == "data" and parts[1] == "chroma_db" and parts[-1] != ".gitkeep":
                continue
            if len(parts) > 2 and parts[0] == "data" and parts[1] == "uploads" and parts[-1] != ".gitkeep":
                continue

            # Write file to zip
            arcname = file_path.relative_to(ROOT_DIR)
            zf.write(file_path, arcname=arcname)
            print(f"Added: {arcname}")

if __name__ == "__main__":
    make_zip(PROJECT_ZIP)
    # Also copy to Desktop root for convenience
    try:
        shutil.copyfile(PROJECT_ZIP, DESKTOP_ZIP)
        print(f"Copied archive to Desktop: {DESKTOP_ZIP}")
    except Exception as e:
        print(f"Desktop copy notice: {e}")
    print("\nZip packaging complete!")
