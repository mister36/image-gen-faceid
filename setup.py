#!/usr/bin/env python3
"""
Setup script for IP-Adapter FaceID Image Generator.
This script handles the installation of IP-Adapter and downloads required models.
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {e.stderr}")
        return None

def clone_ip_adapter():
    """Clone the IP-Adapter repository."""
    logger.info("Cloning IP-Adapter repository...")
    
    if os.path.exists("IP-Adapter"):
        logger.info("IP-Adapter directory already exists, pulling latest changes...")
        run_command("git pull", cwd="IP-Adapter")
    else:
        result = run_command("git clone https://github.com/tencent-ailab/IP-Adapter.git")
        if result is None:
            logger.error("Failed to clone IP-Adapter repository")
            return False
    
    # Install IP-Adapter
    logger.info("Installing IP-Adapter...")
    result = run_command("pip install -e .", cwd="IP-Adapter")
    if result is None:
        logger.error("Failed to install IP-Adapter")
        return False
    
    return True

def download_model(url, filename, directory="models"):
    """Download a model file."""
    Path(directory).mkdir(exist_ok=True)
    filepath = Path(directory) / filename
    
    if filepath.exists():
        logger.info(f"Model {filename} already exists, skipping download")
        return True
    
    logger.info(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {filename} successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False

def download_required_models():
    """Download all required model files."""
    logger.info("Downloading required models...")
    
    models = {
        "ip-adapter-faceid_sd15.bin": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin",
        "ip-adapter-faceid-plus_sd15.bin": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin",
        "ip-adapter-faceid-plusv2_sd15.bin": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
        "ip-adapter-faceid-portrait_sd15.bin": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sd15.bin",
    }
    
    success = True
    for filename, url in models.items():
        if not download_model(url, filename):
            success = False
    
    return success

def setup_environment():
    """Set up the complete environment."""
    logger.info("Setting up IP-Adapter FaceID Image Generator environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Install requirements
    logger.info("Installing Python requirements...")
    result = run_command("pip install -r requirements.txt")
    if result is None:
        logger.error("Failed to install requirements")
        return False
    
    # Clone and install IP-Adapter
    if not clone_ip_adapter():
        return False
    
    # Download models
    if not download_required_models():
        logger.warning("Some models failed to download. They will be downloaded on first use.")
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the server with: python main.py")
    
    return True

if __name__ == "__main__":
    if not setup_environment():
        sys.exit(1)
