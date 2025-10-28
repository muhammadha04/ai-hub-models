#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Simple test script to verify models can be loaded
# ---------------------------------------------------------------------

import sys

print("=" * 70)
print("Testing Combined Segmentation + Depth Detection Models")
print("=" * 70)

# Test 1: Import required modules
print("\n[1/5] Testing imports...")
try:
    import torch
    import numpy as np
    from PIL import Image
    print("✓ Basic dependencies available")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

# Test 2: Import YOLOv11-Segmentation
print("\n[2/5] Testing YOLOv11-Segmentation...")
try:
    from qai_hub_models.models.yolov11_seg.model import YoloV11Segmentor
    print("✓ YOLOv11-Segmentation module available")
    print(f"  Model ID: {YoloV11Segmentor.MODEL_ID}")
except ImportError as e:
    print(f"✗ Cannot import YOLOv11-Seg: {e}")
    print("  Install with: pip install \"qai-hub-models[yolov11-seg]\"")
    sys.exit(1)

# Test 3: Import Depth-Anything-V2
print("\n[3/5] Testing Depth-Anything-V2...")
try:
    from qai_hub_models.models.depth_anything_v2.model import DepthAnythingV2
    print("✓ Depth-Anything-V2 module available")
    print(f"  Model ID: {DepthAnythingV2.MODEL_ID}")
except ImportError as e:
    print(f"✗ Cannot import Depth-Anything-V2: {e}")
    print("  Install with: pip install \"qai-hub-models[depth-anything-v2]\"")
    sys.exit(1)

# Test 4: Check model classes
print("\n[4/5] Checking model information...")
try:
    from qai_hub_models.models.yolov11_seg.model import MODEL_ID as SEG_MODEL_ID
    from qai_hub_models.models.depth_anything_v2.model import MODEL_ID as DEPTH_MODEL_ID
    
    print(f"✓ Segmentation Model: {SEG_MODEL_ID}")
    print(f"✓ Depth Model: {DEPTH_MODEL_ID}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Test model loading (optional, can be slow)
print("\n[5/5] Testing model instantiation...")
print("Note: First time may download pretrained weights (~100MB total)")
print("This may take a few minutes...")

try:
    print("\n  Loading YOLOv11-Segmentation...")
    seg_model = YoloV11Segmentor.from_pretrained()
    print("  ✓ YOLOv11-Segmentation loaded successfully")
    
    print("\n  Loading Depth-Anything-V2...")
    depth_model = DepthAnythingV2.from_pretrained()
    print("  ✓ Depth-Anything-V2 loaded successfully")
    
    print("\n" + "=" * 70)
    print("SUCCESS! All models loaded correctly")
    print("=" * 70)
    print("\nYou can now run:")
    print("  python3 combined_segmentation_depth_demo.py")
    print("\nOr export for on-device deployment:")
    print("  python3 combined_segmentation_depth_export.py --quantize")
    
except Exception as e:
    print(f"\n✗ Error loading models: {e}")
    print("\nThis is expected if:")
    print("  - Models haven't been downloaded yet")
    print("  - Network connection issues")
    print("  - Insufficient memory")
    print("\nTry running the export commands separately:")
    print("  python -m qai_hub_models.models.yolov11_seg.demo")
    print("  python -m qai_hub_models.models.depth_anything_v2.demo")
    sys.exit(1)

print("\n" + "=" * 70)
print("All tests passed!")
print("=" * 70)
