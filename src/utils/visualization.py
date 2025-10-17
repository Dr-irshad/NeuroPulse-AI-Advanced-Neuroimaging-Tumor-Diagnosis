"""
visualization.py
----------------
Provides visualization utilities for displaying tumor segmentation results.
Supports overlay of masks, bounding boxes, and confidence scores on MRI scans.

Author: Dr.Irshad Ibrahim
Date: 2025
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

# Define color palette for visualization
COLORS = {
    "glioma": (255, 0, 0),         # Red
    "meningioma": (0, 255, 0),     # Green
    "pituitary": (0, 128, 255),    # Orange-blue
    "default": (200, 200, 200)     # Gray
}

def apply_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    """
    Apply a semi-transparent mask overlay to an image.

    Args:
        image (np.ndarray): Original RGB image.
        mask (np.ndarray): Binary segmentation mask.
        color (Tuple[int, int, int]): RGB color for the overlay.
        alpha (float): Transparency level (0.0–1.0).

    Returns:
        np.ndarray: Image with mask overlay.
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    mask: np.ndarray = None,
    tumor_type: str = "glioma"
) -> np.ndarray:
    """
    Draw segmentation mask, bounding boxes, and confidence scores on the MRI image.

    Args:
        image (np.ndarray): Input image (RGB).
        boxes (np.ndarray): Bounding box coordinates (x1, y1, x2, y2).
        scores (np.ndarray): Confidence scores for detections.
        classes (np.ndarray): Predicted tumor class indices.
        mask (np.ndarray): Optional segmentation mask.
        tumor_type (str): Tumor class label for color selection.

    Returns:
        np.ndarray: Annotated image.
    """
    output = image.copy()
    color = COLORS.get(tumor_type, COLORS["default"])

    # Overlay mask if provided
    if mask is not None and mask.any():
        output = apply_mask(output, mask, color=color, alpha=0.4)

    # Draw bounding boxes and labels
    for (box, score, cls_id) in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{tumor_type}: {score:.2f}"
        cv2.putText(output, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    print(f"[INFO] Visualization generated with {len(boxes)} detections.")
    return output

def visualize_pipeline_output(result_dict: Dict, original_image: np.ndarray, tumor_type: str) -> np.ndarray:
    """
    Combine segmentation and bounding box visualization for the Streamlit dashboard.

    Args:
        result_dict (Dict): YOLOv8 inference output dictionary.
        original_image (np.ndarray): Original MRI image.
        tumor_type (str): Detected tumor type.

    Returns:
        np.ndarray: Annotated MRI visualization.
    """
    annotated = draw_detections(
        image=original_image,
        boxes=result_dict["boxes"],
        scores=result_dict["scores"],
        classes=result_dict["classes"],
        mask=result_dict["masks"][0] if result_dict["masks"] is not None else None,
        tumor_type=tumor_type
    )
    return annotated

# Example use (comment out when importing as module)
if __name__ == "__main__":
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    dummy_mask = np.zeros((512, 512), dtype=np.uint8)
    dummy_mask[150:300, 200:350] = 255
    dummy_boxes = np.array([[200, 150, 350, 300]])
    dummy_scores = np.array([0.94])
    dummy_classes = np.array([0])

    output_img = draw_detections(dummy_img, dummy_boxes, dummy_scores, dummy_classes, dummy_mask, "glioma")
    cv2.imwrite("visualized_output.png", output_img)
    print("[INFO] Visualization saved as visualized_output.png")

