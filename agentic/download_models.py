#!/usr/bin/env python3
"""
Pre-download all required models for SAM-3D-Objects.
Run this script ONCE with internet connection to cache all models locally.
After that, the app can run in offline mode.
"""

import os
import sys
from pathlib import Path

# Add notebook to path
sys.path.append(str(Path(__file__).parent.parent / "notebook"))

print("🚀 Starting model download...")
print("This may take 10-20 minutes depending on your internet speed.")
print("Models will be cached in ~/.cache/huggingface/")
print()

# Download DINO model (used by SAM-3D condition embedder)
print("📥 Downloading DINO model...")
try:
    from transformers import AutoModel, AutoImageProcessor
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    print("✅ DINO model cached")
except Exception as e:
    print(f"❌ DINO download failed: {e}")

# Download MoGe model (used for depth estimation)
print("\n📥 Downloading MoGe model...")
try:
    from moge.model.v1 import MoGeModel
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
    print("✅ MoGe model cached")
except Exception as e:
    print(f"❌ MoGe download failed: {e}")

# Initialize SAM-3D pipeline (downloads all remaining models)
print("\n📥 Initializing SAM-3D pipeline (downloads all checkpoint files)...")
try:
    from inference import Inference
    config_path = str(Path(__file__).parent.parent / "checkpoints/hf/pipeline.yaml")
    pipeline = Inference(config_path, compile=False)
    print("✅ SAM-3D pipeline models cached")
except Exception as e:
    print(f"❌ SAM-3D initialization failed: {e}")

print("\n" + "="*60)
print("✅ All models downloaded and cached!")
print("="*60)
print()
print("Cache location: ~/.cache/huggingface/")
print()
print("You can now run the app in offline mode:")
print("  python app_gradio_rows.py")
print()
print("The app will use cached models without internet connection.")
