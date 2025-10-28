#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Combined Segmentation + Depth Detection Demo
# ---------------------------------------------------------------------
# This script demonstrates how to combine:
# 1. YOLOv11-Segmentation (quantized) for object segmentation
# 2. Depth-Anything-V2 (quantized) for depth estimation
#
# The output shows detected objects with their segmentation masks and depth information
# ---------------------------------------------------------------------

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.depth_anything_v2.model import DepthAnythingV2
from qai_hub_models.models.yolov11_seg.model import YoloV11Segmentor
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.draw import create_color_map


class CombinedSegmentationDepthApp:
    """
    Application that combines object segmentation with depth estimation.
    
    For a given image input, the app will:
    1. Run YOLOv11 segmentation to detect objects and generate segmentation masks
    2. Run Depth Anything V2 to estimate depth at each point in the image
    3. Combine the results to show segmented objects with their depth information
    """
    
    def __init__(
        self,
        segmentation_model: Optional[YoloV11Segmentor] = None,
        depth_model: Optional[DepthAnythingV2] = None,
        quantize: bool = True,
    ):
        """
        Initialize the combined application.
        
        Parameters:
            segmentation_model: YOLOv11 segmentation model instance
            depth_model: Depth Anything V2 model instance
            quantize: Whether to use quantized models
        """
        print("Initializing Combined Segmentation + Depth Detection App...")
        
        # Load models
        self.quantize = quantize
        
        if segmentation_model is None:
            print("Loading YOLOv11-Segmentation model...")
            self.segmentation_model = YoloV11Segmentor.from_pretrained()
        else:
            self.segmentation_model = segmentation_model
            
        if depth_model is None:
            print("Loading Depth-Anything-V2 model...")
            self.depth_model = DepthAnythingV2.from_pretrained()
        else:
            self.depth_model = depth_model
            
        # Create app wrappers
        from qai_hub_models.models._shared.yolo.app import YoloSegmentationApp
        from qai_hub_models.models._shared.depth_estimation.app import DepthEstimationApp
        
        self.seg_app = YoloSegmentationApp(
            self.segmentation_model,
            nms_score_threshold=0.45,
            nms_iou_threshold=0.7,
            input_height=640,
            input_width=640,
        )
        
        self.depth_app = DepthEstimationApp(
            self.depth_model,
            input_height=518,
            input_width=518,
        )
        
        print("Models loaded successfully!")
        
    def predict(
        self,
        image: Image.Image | str | Path,
        output_path: Optional[str | Path] = None,
    ) -> tuple[Image.Image, dict]:
        """
        Run combined segmentation + depth prediction on an image.
        
        Parameters:
            image: Input image (PIL Image or path to image file)
            output_path: Optional path to save the output visualization
            
        Returns:
            output_image: PIL Image with visualization
            results_dict: Dictionary containing raw results
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = load_image(str(image))
        
        print(f"Processing image of size {image.size}...")
        
        # Run segmentation
        print("Running segmentation...")
        seg_results = self.seg_app.predict_segmentation_from_image(
            image, raw_output=True
        )
        pred_boxes, pred_scores, pred_masks, pred_class_idx = seg_results
        
        # Run depth estimation
        print("Running depth estimation...")
        depth_map = self.depth_app.estimate_depth(image, raw_output=True)
        
        # Combine results and create visualization
        print("Creating combined visualization...")
        output_image = self._create_visualization(
            image, pred_boxes, pred_scores, pred_masks, pred_class_idx, depth_map
        )
        
        # Save if output path provided
        if output_path:
            output_image.save(output_path)
            print(f"Output saved to {output_path}")
        
        # Prepare results dictionary
        results = {
            "num_objects": len(pred_boxes[0]) if pred_boxes else 0,
            "boxes": pred_boxes,
            "scores": pred_scores,
            "masks": pred_masks,
            "classes": pred_class_idx,
            "depth_map": depth_map,
        }
        
        print(f"Detected {results['num_objects']} objects")
        
        return output_image, results
    
    def _create_visualization(
        self,
        image: Image.Image,
        pred_boxes: list,
        pred_scores: list,
        pred_masks: torch.Tensor,
        pred_class_idx: list,
        depth_map: np.ndarray,
    ) -> Image.Image:
        """
        Create a combined visualization showing segmentation masks and depth.
        
        Returns a 2x2 grid showing:
        1. Original image
        2. Segmentation masks
        3. Depth heatmap
        4. Combined segmentation + depth
        """
        # Convert image to numpy
        img_np = np.array(image)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # 1. Original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Segmentation masks
        if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
            # Create colored segmentation mask
            pred_mask_img = torch.argmax(pred_masks, 1)
            color_map = create_color_map(pred_mask_img.max().item() + 1)
            seg_mask_colored = color_map[pred_mask_img[0].numpy()]
            
            # Overlay on original image
            seg_overlay = Image.blend(
                Image.fromarray(img_np),
                Image.fromarray(seg_mask_colored),
                alpha=0.5,
            )
            axes[0, 1].imshow(seg_overlay)
            axes[0, 1].set_title(
                f"Segmentation ({len(pred_boxes[0])} objects detected)",
                fontsize=14,
                fontweight='bold'
            )
        else:
            axes[0, 1].imshow(img_np)
            axes[0, 1].set_title("Segmentation (no objects detected)", fontsize=14)
        axes[0, 1].axis('off')
        
        # 3. Depth heatmap
        depth_normalized = depth_map / depth_map.max()
        depth_colored = plt.cm.plasma(depth_normalized)
        axes[1, 0].imshow(depth_colored)
        axes[1, 0].set_title("Depth Estimation", fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Combined: Segmentation masks with depth
        if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
            # For each detected object, show its depth statistics
            combined = np.array(img_np).copy()
            
            # Overlay segmentation with depth-based coloring
            pred_mask_img = torch.argmax(pred_masks, 1)
            mask_binary = (pred_mask_img[0] > 0).numpy()
            
            # Apply depth coloring to masked regions
            depth_colored_rgb = (plt.cm.plasma(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
            combined[mask_binary] = (
                combined[mask_binary] * 0.4 + depth_colored_rgb[mask_binary] * 0.6
            ).astype(np.uint8)
            
            axes[1, 1].imshow(combined)
            
            # Add depth statistics for each object
            title_parts = ["Combined: Segmentation + Depth\n"]
            for i, (box, score, class_idx) in enumerate(
                zip(pred_boxes[0], pred_scores[0], pred_class_idx[0])
            ):
                if i < 5:  # Show stats for first 5 objects
                    # Get mask for this object
                    obj_mask = (pred_mask_img[0] == i + 1).numpy()
                    if obj_mask.sum() > 0:
                        # Calculate depth statistics for this object
                        obj_depth = depth_map[obj_mask]
                        mean_depth = obj_depth.mean()
                        title_parts.append(
                            f"Obj {i+1} (cls={class_idx.item()}, "
                            f"score={score.item():.2f}): avg_depth={mean_depth:.2f}"
                        )
            
            axes[1, 1].set_title(
                "\n".join(title_parts[:6]),  # Limit to avoid clutter
                fontsize=10,
                fontweight='bold'
            )
        else:
            axes[1, 1].imshow(img_np)
            axes[1, 1].set_title("Combined (no objects to analyze)", fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        output_image = Image.fromarray(img_array)
        
        plt.close(fig)
        
        return output_image


def main():
    parser = argparse.ArgumentParser(
        description="Combined Segmentation + Depth Detection Demo"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image. If not provided, uses a default test image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_output.png",
        help="Path to save output visualization (default: combined_output.png)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Use quantized models (w8a16). Note: Quantization requires AI Hub compilation.",
    )
    
    args = parser.parse_args()
    
    # Load default test image if none provided
    if args.image is None:
        print("No image provided, using default test image...")
        # Use YOLOv11's test image
        from qai_hub_models.models.yolov11_seg.model import (
            MODEL_ASSET_VERSION,
            MODEL_ID,
        )
        image = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "test_images/bus.jpg"
        ).fetch()
        image = load_image(image)
    else:
        image = args.image
    
    # Create and run the combined app
    app = CombinedSegmentationDepthApp(quantize=args.quantize)
    output_image, results = app.predict(image, output_path=args.output)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Number of objects detected: {results['num_objects']}")
    print(f"Depth map range: {results['depth_map'].min():.2f} to {results['depth_map'].max():.2f}")
    print(f"Output saved to: {args.output}")
    print("=" * 60)
    
    # Display the output
    try:
        output_image.show()
    except Exception as e:
        print(f"Could not display image automatically: {e}")
        print(f"Please open {args.output} to view the results.")


if __name__ == "__main__":
    main()
