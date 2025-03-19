import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from inference_sdk import InferenceHTTPClient
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

# Initialize Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="5up3TmaXPrPKv7zh6HBc"
)

# Initialize LangChain Model (DeepSeek via Ollama)
llm_engine = ChatOllama(
    model="deepseek-r1:1.5b",  
    base_url="http://localhost:11434",
    temperature=0.3
)

# Streamlit UI Setup
st.set_page_config(page_title="NeuroPulse AI", layout="wide")

st.sidebar.header("Upload Image or Video File")
uploaded_file = st.sidebar.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg", "mp4", "avi"])

st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0, 100, 50)
overlap_threshold = st.sidebar.slider("Overlap Threshold:", 0, 100, 50)

st.title("🧠 NeuroPulse AI: LLM-Enhanced Neuroimaging & Tumor Diagnosis")
st.write("Upload an MRI scan and detect brain tumors using AI.")

def process_image(image_path, confidence_threshold):
    try:
        # Run inference
        result = CLIENT.infer(image_path, model_id="tumor-otak-tbpou/1")
        
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process predictions
        for pred in result.get("predictions", []):
            x, y, width, height = pred["x"], pred["y"], pred["width"], pred["height"]
            confidence = pred["confidence"]
            label = pred["class"]

            if confidence * 100 >= confidence_threshold:
                # Convert center coordinates to top-left format
                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)

                # Draw bounding box
                color = (255, 0, 255)  # Magenta
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Add label
                text = f"{label}: {confidence*100:.0f}%"
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert back to PIL
        image_pil = Image.fromarray(image)
        return image_pil, result
    
    except Exception as e:
        return None, str(e)

def generate_tumor_description(result):
    if "predictions" not in result or len(result["predictions"]) == 0:
        return "No tumor detected in the MRI scan. If you suspect an issue, please consult a medical professional."
    
    tumor_details = "\n".join([
        f"Tumor detected: {pred['class']} with {pred['confidence']*100:.2f}% confidence."
        for pred in result["predictions"]
    ])
    
    prompt_chain = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a medical AI assistant. Provide insights based on detected tumor results."),
        HumanMessagePromptTemplate.from_template(f"Analyze this tumor detection result and provide insights: {tumor_details}")
    ])
    
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    response = processing_pipeline.invoke({})
    return response

def process_video(video_path, confidence_threshold):
    cap = cv2.VideoCapture(video_path)
    temp_output_path = "processed_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        temp_frame_path = "temp_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)
        processed_frame, _ = process_image(temp_frame_path, confidence_threshold)

        if processed_frame:
            processed_frame_cv = np.array(processed_frame)
            processed_frame_cv = cv2.cvtColor(processed_frame_cv, cv2.COLOR_RGB2BGR)
            out.write(processed_frame_cv)
    
    cap.release()
    out.release()
    return temp_output_path

# Process Uploaded File
if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension in ["jpg", "png", "jpeg"]:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        processed_image, result = process_image("temp_image.jpg", confidence_threshold)
        
        if processed_image:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(processed_image, caption="Detected Tumors", use_container_width=True)
            with col3:
                st.subheader("Detection Details")
                st.json(result)
            
            description = generate_tumor_description(result)
            st.subheader("DeepSeek Medical Assistant")
            st.write(description)
        else:
            st.warning("No tumors detected! Try adjusting the confidence threshold.")
    
    elif file_extension in ["mp4", "avi"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_extension) as temp_video:
            temp_video.write(uploaded_file.getbuffer())
            temp_video_path = temp_video.name
        
        st.video(temp_video_path, format=file_extension)
        
        with st.spinner("Processing video, please wait..."):
            processed_video_path = process_video(temp_video_path, confidence_threshold)
        
        st.video(processed_video_path, format="video/avi")
        st.success("Video processing complete!")
    
    else:
        st.error("Unsupported file format. Please upload an image or a video.")
else:
    st.info("📤 Upload an image or video to start detection!")

