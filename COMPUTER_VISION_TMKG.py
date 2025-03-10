import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
import easyocr

# Set device for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Load models and initialize components
model = resnet18(pretrained=True).to(device)
model.eval()
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

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
        x, y, w, h = max(0, x - 10), max(0, y - 10), min(image.shape[1] - x, w + 20), min(image.shape[0] - y, h + 20)
        return (x, y, w, h), largest_contour
    return None, None

def check_structural_integrity(contour):
    if contour is None:
        return "Unknown", 0.0
    
    hull = cv2.convexHull(contour)
    hull_area, contour_area = cv2.contourArea(hull), cv2.contourArea(contour)
    
    if hull_area == 0:
        return "Unknown", 0.0
    
    solidity = contour_area / hull_area
    bend_confidence = min((0.93 - solidity) * 4, 1.0) if solidity < 0.93 else 0.0
    return ("Bent", bend_confidence) if solidity < 0.93 else ("Straight", solidity)

def detect_pole(image):
    gray, edges = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        vertical_lines = [line[0] for line in lines if abs(line[0][0] - line[0][2]) < 10]
        if vertical_lines:
            avg_x = np.mean([(x1 + x2)/2 for x1, _, x2, _ in vertical_lines])
            return int(avg_x - 10), 0, 20, image.shape[0]
    return None

def detect_tears(image):
    bbox, billboard_contour = detect_billboard_region(image)
    if bbox is None:
        return image, "No", 0.0, "Unknown", 0.0
    
    x, y, w, h = bbox
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [billboard_contour], -1, (255, 255, 255), -1)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    pole_rect = detect_pole(image)
    if pole_rect:
        px, py, pw, ph = pole_rect
        cv2.rectangle(gray_mask, (px, py), (px+pw, py+ph), 0, -1)
    
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=gray_mask)
    
    structure_status, structure_conf = check_structural_integrity(billboard_contour)
    
    return edges, "Yes", 1.0, structure_status, structure_conf

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return " ".join([text[1] for text in reader.readtext(gray)])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    with st.spinner('Analyzing...'):
        tear_debug, torn, tear_conf, structure_status, structure_conf = detect_tears(image_cv)
        text_content = extract_text(image_cv)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(tear_debug, caption='Billboard Analysis')
    
    with col2:
        st.metric("Tear Status", f"{torn} ({tear_conf:.0%} Confidence)")
        st.metric("Structural Integrity", f"{structure_status} ({structure_conf:.0%} Confidence)")
        st.write("**Extracted Text:**", text_content if text_content else "No text found")
    
    compliance_score = max(0, 100 - (tear_conf * 15 + (20 if structure_status == "Bent" else 0)))
    st.subheader(f"Compliance Score: {compliance_score:.0f}/100")
    st.progress(float(compliance_score) / 100)
    
    if compliance_score >= 80 and text_content:
        st.success("✅ Compliant Billboard")
    elif compliance_score >= 50:
        st.warning("⚠️ Needs Improvements")
    else:
        st.error("❌ Non-Compliant")
