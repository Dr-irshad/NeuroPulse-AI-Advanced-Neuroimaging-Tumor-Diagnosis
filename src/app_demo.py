"""
app_demo.py
-----------
Streamlit-based interface to demonstrate the hybrid DL + LLM workflow.
Users can upload an MRI, visualize segmentation, and view LLM-generated output.
"""

import streamlit as st
from model_pipeline import HybridSegmentationPipeline
from llm_integration import DiagnosticReasoner
import numpy as np
import cv2

st.set_page_config(page_title="Hybrid DL + LLM Neuroimaging", layout="wide")
st.title("🧠 Hybrid Deep Learning + LLM Architecture for Medical Imaging")

uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded:
    st.image(uploaded, caption="Uploaded MRI", use_column_width=True)

    # Initialize modules
    seg_pipe = HybridSegmentationPipeline()
    llm_module = DiagnosticReasoner()

    # Run pipeline
    image_array = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    result = seg_pipe.run_inference(image)
    features = seg_pipe.extract_features(result)
    report = llm_module.generate_report(features)

    # Display results
    st.markdown("### 🧩 Extracted Features")
    st.json(features)

    st.markdown("### 🧠 LLM Diagnostic Report")
    st.code(report)
else:
    st.info("Upload an MRI image to begin segmentation and reasoning.")

