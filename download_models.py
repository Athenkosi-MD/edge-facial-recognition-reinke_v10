import os
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must match the path in embedding_service.py
INSIGHTFACE_HOME = os.environ.get('INSIGHTFACE_HOME', '/tmp/insightface')
MODEL_DIR = os.path.join(INSIGHTFACE_HOME, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = {
    "buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
}

def download_and_extract(name, url):
    zip_path = os.path.join(MODEL_DIR, f"{name}.zip")
    extract_path = os.path.join(MODEL_DIR, name)
    
    # Check if already properly extracted (has .onnx files)
    if os.path.isdir(extract_path) and any(f.endswith('.onnx') for f in os.listdir(extract_path)):
        logger.info(f"✓ {name} already exists at {extract_path}, skipping")
        return extract_path
    
    logger.info(f"Downloading {name} to {MODEL_DIR}...")
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract to temp first
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(dir=MODEL_DIR)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmpdir)
    
    # Move .onnx files to correct location
    os.makedirs(extract_path, exist_ok=True)
    onnx_found = False
    for root, dirs, files in os.walk(tmpdir):
        for file in files:
            if file.endswith('.onnx'):
                src = os.path.join(root, file)
                dst = os.path.join(extract_path, file)
                shutil.move(src, dst)
                onnx_found = True
    
    # Cleanup
    os.remove(zip_path)
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    if onnx_found:
        logger.info(f"✓ {name} ready at {extract_path}")
        logger.info(f"  Files: {os.listdir(extract_path)}")
    else:
        raise RuntimeError(f"No .onnx files found in {name} zip!")
    
    return extract_path

if __name__ == "__main__":
    for name, url in MODELS.items():
        download_and_extract(name, url)
    
    # Verify
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Contents: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'NOT FOUND'}")
