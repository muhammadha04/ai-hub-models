# Getting Started: Combined Segmentation + Depth Detection

This guide will help you quickly set up and run a combined **quantized segmentation** and **quantized depth estimation** pipeline using the best models available in Qualcomm AI Hub.

## 📋 Overview

You now have a complete system that combines:
- **YOLOv11-Segmentation** (quantized w8a16) - State-of-the-art object segmentation
- **Depth-Anything-V2** (quantized int8) - Advanced monocular depth estimation

This gives you a detection model that outputs:
- ✅ Object segmentation masks
- ✅ Depth information for each object
- ✅ Spatial relationships between objects

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Install the AI Hub Models package with required dependencies
pip install "qai-hub-models[yolov11-seg,depth-anything-v2]"

# Install visualization dependencies
pip install matplotlib
```

### Step 2: Run the Demo

```bash
# Run with default test image
python3 combined_segmentation_depth_demo.py

# Or use your own image
python3 combined_segmentation_depth_demo.py --image /path/to/your/image.jpg
```

### Step 3: View Results

The script will create a visualization showing:
1. Original image
2. Segmentation masks overlaid
3. Depth heatmap
4. Combined segmentation + depth with statistics

Output is saved to `combined_output.png` by default.

## 🎯 What These Scripts Do

### 1. `combined_segmentation_depth_demo.py`
**Purpose**: Run inference locally using both models

**Features**:
- Loads pretrained YOLOv11-Seg and Depth-Anything-V2
- Processes images through both models
- Creates 2x2 visualization grid
- Calculates depth statistics per detected object

**Usage**:
```bash
python3 combined_segmentation_depth_demo.py \
  --image input.jpg \
  --output result.png
```

### 2. `combined_segmentation_depth_export.py`
**Purpose**: Export quantized models for on-device deployment

**Features**:
- Shows commands to export models to Qualcomm devices
- Supports quantization (w8a16 for YOLO, int8 for Depth)
- Profiles performance on real devices
- Exports to TFLite, QNN, or ONNX formats

**Usage**:
```bash
# Show export commands
python3 combined_segmentation_depth_export.py --quantize

# Export for specific device
python3 combined_segmentation_depth_export.py \
  --quantize \
  --device "Samsung Galaxy S24" \
  --profile
```

### 3. `test_combined_models.py`
**Purpose**: Verify your environment is set up correctly

**Usage**:
```bash
python3 test_combined_models.py
```

## 📦 Files Created

```
/workspace/
├── combined_segmentation_depth_demo.py    # Main demo script
├── combined_segmentation_depth_export.py  # Export helper script
├── test_combined_models.py                # Environment test script
├── COMBINED_MODELS_README.md              # Detailed documentation
└── GETTING_STARTED.md                     # This file
```

## 🔧 Advanced: Export for On-Device Deployment

### Prerequisites
1. Sign up at [Qualcomm AI Hub](https://aihub.qualcomm.com/)
2. Get your API token from [your account page](https://app.aihub.qualcomm.com/account/)
3. Configure the SDK:
   ```bash
   pip install qai-hub
   qai-hub configure --api_token YOUR_API_TOKEN
   ```

### Export YOLOv11-Segmentation (Quantized)

```bash
python -m qai_hub_models.models.yolov11_seg.export \
  --quantize w8a16 \
  --device "Samsung Galaxy S24" \
  --target-runtime tflite \
  --profile
```

**What this does**:
- Converts PyTorch model to ONNX
- Quantizes to 8-bit weights, 16-bit activations (w8a16)
- Compiles for Snapdragon device
- Profiles inference time on real device
- Downloads compiled model

**Expected Performance** (Snapdragon 8 Gen 3):
- Inference: ~20-30ms
- Model size: ~11 MB
- Accuracy: mIoU ~40+ (COCO dataset)

### Export Depth-Anything-V2 (Quantized)

```bash
python -m qai_hub_models.models.depth_anything_v2.export \
  --device "Samsung Galaxy S24" \
  --target-runtime tflite \
  --profile
```

**What this does**:
- Converts to TFLite format
- Automatic int8 quantization
- Compiles for NPU acceleration
- Profiles on real device

**Expected Performance** (Snapdragon 8 Gen 3):
- Inference: ~100-150ms
- Model size: ~25 MB (quantized)
- Accuracy: High quality depth maps

## 🎨 Example Use Cases

### 1. Robotics Navigation
```python
from combined_segmentation_depth_demo import CombinedSegmentationDepthApp

app = CombinedSegmentationDepthApp()
output, results = app.predict("robot_view.jpg")

# Check if path is clear
for i, box in enumerate(results['boxes'][0]):
    obj_depth = calculate_object_depth(results, i)
    if obj_depth < 1.0:  # Object is less than 1 meter away
        print(f"Obstacle detected: {obj_depth:.2f}m")
```

### 2. AR Object Placement
```python
# Get depth at specific pixel location
depth_at_tap = results['depth_map'][tap_y, tap_x]
print(f"Place AR object at depth: {depth_at_tap:.2f}")
```

### 3. Safety Monitoring
```python
# Check if person is too close to machinery
for i, cls_idx in enumerate(results['classes'][0]):
    if cls_idx == 0:  # Person class in COCO
        person_depth = calculate_object_depth(results, i)
        if person_depth < SAFE_DISTANCE:
            trigger_alert()
```

## 📊 Model Information

### YOLOv11-Segmentation
- **Link**: [aihub.qualcomm.com/models/yolov11_seg](https://aihub.qualcomm.com/models/yolov11_seg)
- **Classes**: 80 COCO classes (person, car, dog, etc.)
- **Input**: 640×640 RGB
- **Output**: Boxes + Masks + Classes
- **Quantization**: w8a16 (8-bit weights, 16-bit activations)
- **Size**: 11.4 MB (quantized)

### Depth-Anything-V2
- **Link**: [aihub.qualcomm.com/models/depth_anything_v2](https://aihub.qualcomm.com/models/depth_anything_v2)
- **Task**: Monocular depth estimation
- **Input**: 518×518 RGB
- **Output**: Dense depth map
- **Quantization**: int8
- **Size**: ~25 MB (quantized)

## 🛠️ Troubleshooting

### Issue: "No module named 'torch'"
```bash
pip install torch torchvision
```

### Issue: "No module named 'matplotlib'"
```bash
pip install matplotlib
```

### Issue: "No module named 'qai_hub_models'"
```bash
pip install qai-hub-models
```

### Issue: Models download slowly
- Models are ~100MB total
- First run downloads from Qualcomm servers
- Subsequent runs use cached models
- Check internet connection

### Issue: Out of memory
- Close other applications
- Use smaller images
- Consider using quantized models on device instead

### Issue: "AI Hub not configured"
```bash
# Get API token from https://app.aihub.qualcomm.com/account/
qai-hub configure --api_token YOUR_TOKEN
```

## 📱 Supported Devices

The quantized models are optimized for:

**Mobile**:
- Samsung Galaxy S24, S23, S22 (all variants)
- Xiaomi 13, 12 series
- OnePlus devices with Snapdragon 8 series

**Compute**:
- Snapdragon X Elite laptops
- Windows on ARM devices

**IoT**:
- QCS6490, QCS8250, QCS8550
- RB5, RB3 Gen 2

**Automotive**:
- SA8255P, SA8295P, SA8650P, SA8775P

**XR**:
- QCS8450

For complete device list: [aihub.qualcomm.com/devices](https://aihub.qualcomm.com/devices)

## 🔗 Resources

- **Full Documentation**: See `COMBINED_MODELS_README.md`
- **AI Hub Models Repo**: [github.com/quic/ai-hub-models](https://github.com/quic/ai-hub-models)
- **AI Hub Portal**: [aihub.qualcomm.com](https://aihub.qualcomm.com)
- **Sample Apps**: [github.com/quic/ai-hub-apps](https://github.com/quic/ai-hub-apps)
- **Community Slack**: [aihub.qualcomm.com/community/slack](https://aihub.qualcomm.com/community/slack)

## 📧 Support

- **Email**: ai-hub-support@qti.qualcomm.com
- **GitHub Issues**: [ai-hub-models/issues](https://github.com/quic/ai-hub-models/issues)
- **Slack**: Join the AI Hub Community

## ⚖️ License

- **Scripts**: BSD-3-Clause
- **YOLOv11**: AGPL-3.0
- **Depth-Anything-V2**: Apache-2.0

---

**Next Steps**:
1. ✅ Install dependencies: `pip install "qai-hub-models[yolov11-seg,depth-anything-v2]" matplotlib`
2. ✅ Run demo: `python3 combined_segmentation_depth_demo.py`
3. ✅ Read full docs: `COMBINED_MODELS_README.md`
4. ✅ Export for device: `python3 combined_segmentation_depth_export.py --quantize`

Happy coding! 🚀
