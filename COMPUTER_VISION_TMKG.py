import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disable problematic watcher
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Explicitly disable GPU

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights
import easyocr

@st.cache_resource
def load_models():
    # Force CPU-only operations
    device = torch.device('cpu')
    torch.set_default_device(device)
    
    # Initialize ResNet with progress
    with st.spinner('Loading vision model...'):
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights).to(device).eval()
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            st.stop()

# Initialize EasyOCR with progress
    with st.spinner('Preparing OCR engine...'):
        try:
            os.makedirs("models", exist_ok=True)
            reader = easyocr.Reader(
                ['en'],
                gpu=False,
                download_enabled=True,
                model_storage_directory='models',
                recog_network='english_g2'  # Smaller model
            )
        except Exception as e:
            st.error(f"OCR setup failed: {str(e)}")
            st.stop()

    return model, reader

# Initialize models at app start
model, reader = load_models()

# Streamlit UI with enhanced constraints
st.title("TMKG Billboard Compliance Checker")
st.write("Upload an image of a billboard to check compliance.")
uploaded_file = st.file_uploader("Choose an image...", 
                               type=["jpg", "jpeg", "png"],
                               accept_multiple_files=False)

# ---------- Keep original functions unchanged ----------
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

def check_structural_integrity(contour):
    if contour is None:
        return "Unknown", 0.0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    
    if hull_area == 0:
        return "Unknown", 0.0
    
    solidity = contour_area / hull_area
    if solidity < 0.93:
        bend_confidence = min((0.93 - solidity) * 4, 1.0)
        return "Bent", bend_confidence
    return "Straight", solidity

def detect_pole(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    pole_rect = None
    if lines is not None:
        vertical_lines = [line[0] for line in lines if abs(line[0][0] - line[0][2]) < 10]
        if vertical_lines:
            avg_x = np.mean([(x1 + x2)/2 for x1, _, x2, _ in vertical_lines])
            pole_rect = (int(avg_x - 10), 0, 20, image.shape[0])
    return pole_rect

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
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=gray_mask)
    
    border_mask = np.zeros_like(edges)
    cv2.rectangle(border_mask, (x+10, y+10), (x+w-10, y+h-10), 255, -1)
    border_mask = cv2.bitwise_not(border_mask)
    edge_areas = cv2.bitwise_and(edges, edges, mask=border_mask)
    
    contours, _ = cv2.findContours(edge_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tear_detected = False
    max_confidence = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
            
        xc, yc, wc, hc = cv2.boundingRect(c)
        if xc == 0 or yc == 0 or xc+wc == image.shape[1] or yc+hc == image.shape[0]:
            continue
            
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        if circularity > 0.25:
            continue
            
        confidence = min(area / 3000, 1.0)
        max_confidence = max(max_confidence, confidence)
        tear_detected = True
    
    debug_img = cv2.cvtColor(edge_areas, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if pole_rect:
        cv2.rectangle(debug_img, (pole_rect[0], pole_rect[1]),
                    (pole_rect[0]+pole_rect[2], pole_rect[1]+pole_rect[3]),
                    (255, 0, 0), 2)
    
    structure_status, structure_conf = check_structural_integrity(billboard_contour)
    cv2.putText(debug_img, f"Structure: {structure_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return debug_img, "Yes" if tear_detected else "No", max_confidence, structure_status, structure_conf

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

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return " ".join([text[1] for text in reader.readtext(gray)])

# ---------- Updated processing block ----------
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
