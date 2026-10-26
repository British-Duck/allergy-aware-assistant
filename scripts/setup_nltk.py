import os
import zipfile
import nltk

# Always install NLTK data inside the project folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
TARGET_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
CORPORA_DIR = os.path.join(TARGET_DIR, "corpora")
os.makedirs(CORPORA_DIR, exist_ok=True)

# Force NLTK to use this folder FIRST
nltk.data.path = [TARGET_DIR] + [p for p in nltk.data.path if p != TARGET_DIR]

PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]

print("Installing NLTK resources into:", TARGET_DIR)
for pkg in PACKAGES:
    print(f"Downloading {pkg}...")
    nltk.download(pkg, download_dir=TARGET_DIR, force=True, quiet=False)

def extract_if_needed(zip_name: str, folder_name: str):
    """
    Some NLTK corpora may download as .zip but not extract automatically.
    This function ensures the extracted folder exists.
    """
    zip_path = os.path.join(CORPORA_DIR, zip_name)
    folder_path = os.path.join(CORPORA_DIR, folder_name)

    if os.path.isdir(folder_path):
        print(f"OK: {folder_name} already extracted.")
        return

    if not os.path.exists(zip_path):
        print(f"NOTE: {zip_name} not found (nothing to extract).")
        return

    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(CORPORA_DIR)

    if os.path.isdir(folder_path):
        print(f"OK: Extracted {folder_name}.")
    else:
        print(f"WARNING: Expected folder not created: {folder_path}")

# Ensure zipped corpora are extracted
extract_if_needed("wordnet.zip", "wordnet")
extract_if_needed("omw-1.4.zip", "omw-1.4")

print("\n=== Verification (must not error) ===")
print("Verify punkt:", nltk.data.find("tokenizers/punkt"))
print("Verify stopwords:", nltk.data.find("corpora/stopwords"))
print("Verify wordnet:", nltk.data.find("corpora/wordnet"))
print("Verify omw-1.4:", nltk.data.find("corpora/omw-1.4"))

print("\nNLTK resources installed and verified successfully.")
