import sys
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights
import easyocr

# Disable Streamlit's file watcher (prevents conflicts with Torch)
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@st.cache_resource
def load_models():
    device = torch.device('cpu')
    torch.set_default_device(device)

    # Load ResNet model
    with st.spinner('🔧 Loading vision model (1.2GB)...'):
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights).to(device).eval()
        except Exception as e:
            st.error(f"Vision model failed: {str(e)}")
            st.stop()
    
    # Initialize EasyOCR reader
    # Initialize EasyOCR reader (Lightweight Version)
with st.spinner('🔍 Initializing OCR engine...'):
    try:
        reader = easyocr.Reader(
            ['en'], 
            gpu=False, 
            download_enabled=False,  # Prevent unnecessary downloads
            model_storage_directory='models',  
            detector=False,  # Disables text detection, making it faster
            verbose=False  # Prevents excessive logging
        )
    except Exception as e:
        st.error(f"OCR failed: {str(e)}")
        st.stop()

st.title("TMKG Billboard Compliance Checker")
st.write("Upload an image of a billboard to check compliance.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Placeholder functions for missing implementations
def detect_tears(image):
    return image, "No", 0.0, "Stable", 1.0  # Debug Image, Torn Status, Confidence, Structure Status, Confidence

def detect_obstruction(image):
    return image, "No", 0.0  # Obstruction Mask, Obstructed Status, Obstruct Ratio

def check_alignment(image):
    return 1.0  # Alignment Confidence

def analyze_brightness(image):
    return 1.0  # Brightness Score

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return " ".join([text[1] for text in reader.readtext(gray)])

if uploaded_file is not None:
    try:
        if uploaded_file.size > 5_000_000:
            st.error("File size exceeds 5MB limit")
            st.stop()
            
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        with st.spinner('Analyzing...'):
            tear_debug, torn, tear_conf, structure_status, structure_conf = detect_tears(image_cv)
            obstruct_mask, obstructed, obstruct_ratio = detect_obstruction(image_cv)
            align_conf = check_alignment(image_cv)
            brightness = analyze_brightness(image_cv)
            text_content = extract_text(image_cv)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(tear_debug, 
                    caption='Billboard Analysis: Green=Boundary, Blue=Pole, Yellow=Structure Status')
            st.image(obstruct_mask, caption=f'Obstruction: {obstructed} ({obstruct_ratio:.1%})')
        
        with col2:
            st.metric("Tear Status", f"{torn} ({tear_conf:.0%} Confidence)")
            st.metric("Structural Integrity", f"{structure_status} ({structure_conf:.0%} Confidence)")
            st.metric("Alignment", f"{align_conf:.0%} Confidence")
            st.metric("Brightness", f"{brightness:.0%} of Optimal")
            st.write("**Extracted Text:**", text_content if text_content else "No text found")
        
        penalties = {
            'Tear': tear_conf * 15,
            'Obstruction': 10 if obstructed == "Yes" else 0,
            'Misalignment': (1 - align_conf) * 10,
            'Low Brightness': max(0, (0.4 - brightness)) * 15,
            'Structural Damage': structure_conf * 20 if structure_status == "Bent" else 0
        }
        
        compliance_score = max(0, 100 - sum(penalties.values()))
        st.subheader(f"Compliance Score: {compliance_score:.0f}/100")
        st.progress(float(compliance_score) / 100)
        
        with st.expander("Penalty Breakdown"):
            for k, v in penalties.items():
                st.write(f"{k}: -{v:.1f} pts")
        
        if compliance_score >= 80 and text_content:
            st.success("✅ Compliant Billboard")
        elif compliance_score >= 50:
            st.warning("⚠️ Needs Improvements")
        else:
            st.error("❌ Non-Compliant")
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.stop()
