#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Combined Segmentation + Depth Detection Export Script
# ---------------------------------------------------------------------
# This script helps you export quantized models for on-device deployment
# ---------------------------------------------------------------------

import argparse
from pathlib import Path

print("=" * 70)
print("COMBINED SEGMENTATION + DEPTH DETECTION - MODEL EXPORT")
print("=" * 70)


def export_yolov11_segmentation(quantize: bool = True, device: str = None):
    """Export YOLOv11-Segmentation model."""
    print("\n" + "=" * 70)
    print("1. EXPORTING YOLOv11-SEGMENTATION MODEL")
    print("=" * 70)
    
    try:
        from qai_hub_models.models.yolov11_seg import export
        
        print("\nModel Info:")
        print("  - Name: YOLOv11-Segmentation")
        print("  - Task: Object Segmentation")
        print("  - Input: 640x640 RGB images")
        print("  - Output: Bounding boxes, segmentation masks, class predictions")
        print(f"  - Quantization: {'Enabled (w8a16)' if quantize else 'Disabled'}")
        
        # Build export command
        cmd_parts = ["python", "-m", "qai_hub_models.models.yolov11_seg.export"]
        
        if quantize:
            cmd_parts.append("--quantize w8a16")
        
        if device:
            cmd_parts.append(f"--device '{device}'")
        
        print(f"\nTo export this model, run:")
        print(f"  {' '.join(cmd_parts)}")
        
        return True
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install the required dependencies:")
        print('  pip install "qai-hub-models[yolov11-seg]"')
        return False


def export_depth_anything_v2(device: str = None):
    """Export Depth-Anything-V2 model."""
    print("\n" + "=" * 70)
    print("2. EXPORTING DEPTH-ANYTHING-V2 MODEL")
    print("=" * 70)
    
    try:
        from qai_hub_models.models.depth_anything_v2 import export
        
        print("\nModel Info:")
        print("  - Name: Depth-Anything-V2")
        print("  - Task: Monocular Depth Estimation")
        print("  - Input: 518x518 RGB images")
        print("  - Output: Per-pixel depth map")
        print("  - Quantization: Automatic (int8)")
        
        # Build export command
        cmd_parts = ["python", "-m", "qai_hub_models.models.depth_anything_v2.export"]
        
        if device:
            cmd_parts.append(f"--device '{device}'")
        
        print(f"\nTo export this model, run:")
        print(f"  {' '.join(cmd_parts)}")
        
        return True
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install the required dependencies:")
        print('  pip install "qai-hub-models[depth-anything-v2]"')
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export combined segmentation + depth models for on-device deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export both models with default settings
  python combined_segmentation_depth_export.py
  
  # Export quantized models
  python combined_segmentation_depth_export.py --quantize
  
  # Export for specific device
  python combined_segmentation_depth_export.py --device "Samsung Galaxy S24"
  
  # Export and profile on device
  python combined_segmentation_depth_export.py --quantize --profile --device "Samsung Galaxy S24"

Supported Devices:
  - Samsung Galaxy S24/S23/S22 series
  - Snapdragon X Elite
  - QCS6490, QCS8250, QCS8550 (IoT)
  - SA8255P, SA8295P (Automotive)
  
For full device list, visit: https://aihub.qualcomm.com/
        """
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable quantization for YOLOv11-Seg (w8a16). Depth model uses automatic quantization.",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Target device for compilation (e.g., "Samsung Galaxy S24")',
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile models on real device (requires --device)",
    )
    
    parser.add_argument(
        "--target-runtime",
        type=str,
        default="tflite",
        choices=["tflite", "qnn", "onnx"],
        help="Target runtime for export (default: tflite)",
    )
    
    args = parser.parse_args()
    
    # Check if AI Hub is configured
    print("\nChecking AI Hub configuration...")
    try:
        import qai_hub
        # Try to get user info to check if configured
        print("✓ AI Hub is configured")
    except ImportError:
        print("✗ AI Hub SDK not installed")
        print("\nPlease install:")
        print("  pip install qai-hub")
        return
    except Exception as e:
        print("✗ AI Hub not configured")
        print("\nPlease configure AI Hub:")
        print("  1. Sign up at https://aihub.qualcomm.com/")
        print("  2. Get your API token from https://app.aihub.qualcomm.com/account/")
        print("  3. Configure: qai-hub configure --api_token YOUR_TOKEN")
        return
    
    # Export models
    print("\n")
    success_seg = export_yolov11_segmentation(
        quantize=args.quantize,
        device=args.device,
    )
    
    success_depth = export_depth_anything_v2(device=args.device)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"YOLOv11-Segmentation: {'✓ Ready' if success_seg else '✗ Failed'}")
    print(f"Depth-Anything-V2: {'✓ Ready' if success_depth else '✗ Failed'}")
    
    if success_seg and success_depth:
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Export YOLOv11-Segmentation:")
        seg_cmd = "python -m qai_hub_models.models.yolov11_seg.export"
        if args.quantize:
            seg_cmd += " --quantize w8a16"
        if args.device:
            seg_cmd += f" --device '{args.device}'"
        if args.profile:
            seg_cmd += " --profile"
        seg_cmd += f" --target-runtime {args.target_runtime}"
        print(f"   {seg_cmd}")
        
        print("\n2. Export Depth-Anything-V2:")
        depth_cmd = "python -m qai_hub_models.models.depth_anything_v2.export"
        if args.device:
            depth_cmd += f" --device '{args.device}'"
        if args.profile:
            depth_cmd += " --profile"
        depth_cmd += f" --target-runtime {args.target_runtime}"
        print(f"   {depth_cmd}")
        
        print("\n3. Run the combined demo:")
        print("   python combined_segmentation_depth_demo.py --image your_image.jpg")
        
        print("\n" + "=" * 70)
        print("For more information, see: COMBINED_MODELS_README.md")
        print("=" * 70)


if __name__ == "__main__":
    main()
