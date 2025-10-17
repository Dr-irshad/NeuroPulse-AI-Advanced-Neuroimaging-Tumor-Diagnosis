"""
model_pipeline.py
-----------------
Implements the hybrid Deep Learning pipeline for MRI-based tumor segmentation.
This script outlines preprocessing, YOLOv8 inference, and feature extraction for downstream LLM-based medical reasoning.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO

class HybridSegmentationPipeline:
    """
    YOLOv8-Attention–based tumor segmentation and feature extraction pipeline.
    """

    def __init__(self, model_path: str = "weights/yolov8_attention.pt"):
        # Load YOLOv8 model (segmentation variant)
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Model loaded on {self.device}: {model_path}")

    def preprocess_image(self, img_path: str) -> np.ndarray:
        """
        Load and normalize an MRI image.
        """
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"[INFO] Loaded MRI image: {img_path}")
        return image

    def run_inference(self, image: np.ndarray) -> dict:
        """
        Perform segmentation inference using YOLOv8.
        """
        results = self.model.predict(source=image, device=self.device, verbose=False)
        if not results:
            raise ValueError("No detections found.")
        output = results[0]
        print(f"[INFO] Inference complete. {len(output.boxes)} objects detected.")
        return {
            "masks": output.masks.data.cpu().numpy() if output.masks else None,
            "boxes": output.boxes.xyxy.cpu().numpy(),
            "scores": output.boxes.conf.cpu().numpy(),
            "classes": output.boxes.cls.cpu().numpy()
        }

    def extract_features(self, result_dict: dict) -> dict:
        """
        Derive structured metadata from YOLO output for LLM reasoning.
        """
        tumor_area = np.sum(result_dict["masks"]) if result_dict["masks"] is not None else 0
        conf_mean = float(np.mean(result_dict["scores"])) if len(result_dict["scores"]) else 0.0
        print(f"[INFO] Extracted features: area={tumor_area}, confidence={conf_mean:.3f}")

        return {
            "detected": True,
            "mean_confidence": conf_mean,
            "tumor_area_pixels": tumor_area,
            "tumor_class": "glioma" if conf_mean > 0.8 else "meningioma",
            "inference_device": self.device
        }

if __name__ == "__main__":
    pipe = HybridSegmentationPipeline()
    img = pipe.preprocess_image("sample_mri.jpg")
    result = pipe.run_inference(img)
    features = pipe.extract_features(result)
    print(features)

