# Combined Segmentation + Depth Detection

This guide demonstrates how to combine **quantized segmentation** and **quantized depth estimation** models from Qualcomm AI Hub to create a powerful detection system that outputs both object segmentation masks and depth information.

## Models Used

### 1. YOLOv11-Segmentation (Quantized)
- **Purpose**: Real-time object segmentation
- **Output**: Bounding boxes, segmentation masks, and object classes
- **Quantization**: Supports w8a16 quantization
- **Input**: 640x640 RGB images
- **Details**: [YOLOv11-Seg on AI Hub](https://aihub.qualcomm.com/models/yolov11_seg)

### 2. Depth-Anything-V2 (Quantized)
- **Purpose**: Monocular depth estimation
- **Output**: Per-pixel depth map
- **Quantization**: Supports int8 quantization
- **Input**: 518x518 RGB images
- **Details**: [Depth-Anything-V2 on AI Hub](https://aihub.qualcomm.com/models/depth_anything_v2)

## Installation

```bash
# Install the AI Hub Models package with required dependencies
pip install "qai-hub-models[yolov11-seg,depth-anything-v2]"

# Additional dependencies for visualization
pip install matplotlib
```

## Quick Start

### Basic Usage (Local Inference)

Run the combined demo with a default test image:

```bash
python combined_segmentation_depth_demo.py
```

### Using Your Own Image

```bash
python combined_segmentation_depth_demo.py --image /path/to/your/image.jpg --output result.png
```

### Using Quantized Models (Requires AI Hub)

To use quantized models, you need to export them first using AI Hub:

```bash
# Configure AI Hub (required for quantization)
qai-hub configure --api_token YOUR_API_TOKEN

# Export quantized models
python combined_segmentation_depth_export.py --quantize
```

## Output

The script generates a **2x2 visualization grid** showing:

1. **Top-Left**: Original input image
2. **Top-Right**: Segmentation masks overlaid on the image with detected objects
3. **Bottom-Left**: Depth heatmap (warmer colors = closer, cooler colors = farther)
4. **Bottom-Right**: Combined view with segmentation masks colored by depth + depth statistics per object

### Sample Output Information

For each detected object, the system provides:
- Bounding box coordinates
- Segmentation mask
- Object class
- Confidence score
- **Average depth** of the object

## Use Cases

This combined approach is valuable for:

1. **Robotics & Autonomous Navigation**
   - Identify objects and their spatial distance
   - Path planning with obstacle avoidance
   - Grasp planning for robotic arms

2. **Augmented Reality (AR)**
   - Occlusion handling based on depth
   - Object placement in 3D space
   - Scene understanding

3. **Safety & Surveillance**
   - Detect objects and estimate their distance from camera
   - Intrusion detection with distance thresholds
   - Personal protective equipment (PPE) detection with spatial context

4. **Automotive**
   - Pedestrian detection with distance estimation
   - Lane detection with depth awareness
   - Parking assistance

## Advanced Usage

### Export Models for On-Device Deployment

To compile and quantize models for deployment on Qualcomm devices:

```bash
# Export YOLOv11-Segmentation (quantized)
python -m qai_hub_models.models.yolov11_seg.export \
    --quantize w8a16 \
    --device "Samsung Galaxy S24" \
    --target-runtime tflite

# Export Depth-Anything-V2 (quantized)
python -m qai_hub_models.models.depth_anything_v2.export \
    --device "Samsung Galaxy S24" \
    --target-runtime tflite
```

### Profile Models on Real Devices

Check performance on actual Qualcomm devices:

```bash
# Profile YOLOv11-Seg
python -m qai_hub_models.models.yolov11_seg.export \
    --quantize w8a16 \
    --device "Samsung Galaxy S24" \
    --profile

# Profile Depth-Anything-V2
python -m qai_hub_models.models.depth_anything_v2.export \
    --device "Samsung Galaxy S24" \
    --profile
```

## Supported Devices

Both models are optimized for Qualcomm devices including:

- **Mobile**: Samsung Galaxy S24/S23/S22 series, Xiaomi 12/13
- **IoT**: QCS6490, QCS8250, QCS8550
- **Automotive**: SA8255P, SA8295P, SA8650P, SA8775P
- **Compute**: Snapdragon X Elite

See [Qualcomm AI Hub](https://aihub.qualcomm.com) for the complete list of supported devices.

## Performance Notes

### YOLOv11-Segmentation
- **Model Size**: ~11 MB (quantized)
- **Parameters**: 2.89M
- **Inference Time**: ~20-30ms on Snapdragon 8 Gen 3

### Depth-Anything-V2
- **Model Size**: ~94 MB (float), ~25 MB (quantized)
- **Parameters**: 24.7M
- **Inference Time**: ~100-150ms on Snapdragon 8 Gen 3

### Combined Pipeline
- **Total Inference Time**: ~120-180ms (can run in parallel on separate cores)
- **Memory**: ~150 MB total

## Architecture Details

### Pipeline Flow

```
Input Image (RGB)
    |
    |-----> [Resize to 640x640] --> [YOLOv11-Seg] --> Segmentation Masks + Boxes
    |                                                  |
    |-----> [Resize to 518x518] --> [Depth-Anything-V2] --> Depth Map
    |                                                  |
    |                                                  v
    +---------------------------------> [Combine & Visualize]
                                                     |
                                                     v
                                            Final Output:
                                            - Segmented objects
                                            - Depth per object
                                            - Spatial relationships
```

## API Reference

### CombinedSegmentationDepthApp

```python
from combined_segmentation_depth_demo import CombinedSegmentationDepthApp

# Initialize the app
app = CombinedSegmentationDepthApp(quantize=False)

# Run prediction
output_image, results = app.predict(
    image="path/to/image.jpg",
    output_path="output.png"
)

# Access results
print(f"Detected objects: {results['num_objects']}")
print(f"Boxes: {results['boxes']}")
print(f"Masks: {results['masks']}")
print(f"Depth map shape: {results['depth_map'].shape}")
```

## Troubleshooting

### Issue: Models take too long to load
**Solution**: Models are downloaded on first use. Subsequent runs will be faster.

### Issue: Out of memory
**Solution**: 
- Use quantized models
- Process smaller images
- Ensure sufficient RAM (recommended: 8GB+)

### Issue: Quantization not working
**Solution**: 
- Ensure you have configured AI Hub: `qai-hub configure --api_token YOUR_TOKEN`
- Sign up at [AI Hub](https://aihub.qualcomm.com) to get an API token

### Issue: CUDA/GPU errors
**Solution**: The models run on CPU by default. For GPU acceleration, ensure PyTorch with CUDA support is installed.

## References

- [Qualcomm AI Hub Models Repository](https://github.com/quic/ai-hub-models)
- [Qualcomm AI Hub Documentation](https://app.aihub.qualcomm.com/docs/)
- [YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Depth Anything V2 Paper](https://arxiv.org/abs/2406.09414)

## License

- **YOLOv11-Segmentation**: AGPL-3.0
- **Depth-Anything-V2**: Apache-2.0
- **This Script**: BSD-3-Clause (Qualcomm AI Hub Models License)

## Support

- **Slack**: [AI Hub Community](https://aihub.qualcomm.com/community/slack)
- **Email**: ai-hub-support@qti.qualcomm.com
- **GitHub Issues**: [ai-hub-models/issues](https://github.com/quic/ai-hub-models/issues)
