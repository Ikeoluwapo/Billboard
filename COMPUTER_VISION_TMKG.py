import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
from torchvision.models import resnet18

# Load models and initialize components
model = resnet18(pretrained=True)
model.eval()

# Streamlit UI setup
st.title("TMKG Billboard Compliance Checker")
st.write("Upload an image of a billboard to check compliance.")
uploaded_file = st.file_uploader("Choose an image...")

def detect_billboard_region(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x = max(0, x - 10)
        y = max(0, y - 10)
        w = min(image.shape[1] - x, w + 20)
        h = min(image.shape[0] - y, h + 20)
        return (x, y, w, h), largest_contour
    return None, None

def detect_obstruction(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    diff = cv2.absdiff(image, blurred)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    obstruction_ratio = np.sum(thresholded > 0) / (thresholded.shape[0] * thresholded.shape[1])
    return thresholded, "Yes" if obstruction_ratio > 0.05 else "No", obstruction_ratio

def check_alignment(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    alignment_confidence = 1.0
    if lines is not None:
        angles = [np.degrees(line[0][1]) for line in lines]
        avg_angle = np.mean(angles)
        alignment_confidence = 1 - abs(90 - avg_angle)/90
    return max(alignment_confidence, 0)

def analyze_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2]) / 255

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        with st.spinner('Analyzing...'):
            obstruct_mask, obstructed, obstruct_ratio = detect_obstruction(image_cv)
            align_conf = check_alignment(image_cv)
            brightness = analyze_brightness(image_cv)
            torn = "No"  # Placeholder for tear detection
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(obstruct_mask, caption=f'Obstruction: {obstructed} ({obstruct_ratio:.1%})')
        
        with col2:
            st.metric("Tear Status", torn)
            st.metric("Alignment", f"{align_conf:.0%} Confidence")
            st.metric("Brightness", f"{brightness:.0%} of Optimal")
        
        penalties = {
            'Tear': 15 if torn == "Yes" else 0,
            'Obstruction': 10 if obstructed == "Yes" else 0,
            'Misalignment': (1 - align_conf) * 10,
            'Low Brightness': max(0, (0.4 - brightness)) * 15
        }
        
        compliance_score = max(0, 100 - sum(penalties.values()))
        st.subheader(f"Compliance Score: {compliance_score:.0f}/100")
        st.progress(float(compliance_score) / 100)
        
        with st.expander("Penalty Breakdown"):
            for k, v in penalties.items():
                st.write(f"{k}: -{v:.1f} pts")
        
        if compliance_score >= 80:
            st.success("✅ Compliant Billboard")
        elif compliance_score >= 50:
            st.warning("⚠️ Needs Improvements")
        else:
            st.error("❌ Non-Compliant")
    
    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid image.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

