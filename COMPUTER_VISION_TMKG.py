import pytesseract
import easyocr
import cv2
import numpy as np
import streamlit as st
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.write(st.secrets)

# --------- TESSERACT SETUP ---------
if "tesseract_path" in st.secrets:
    pytesseract.pytesseract.tesseract_cmd = st.secrets["tesseract_path"]
else:
    st.warning("⚠️ Tesseract path not found in secrets. Using default system path.")

# Initialize session variables
if "use_easyocr" not in st.session_state:
    st.session_state.use_easyocr = True
if "ocr_reader" not in st.session_state:
    st.session_state.ocr_reader = None

# --------- LOAD OCR MODELS (CACHED) ---------
@st.cache_resource
def load_ocr():
    try:
        reader = easyocr.Reader(
            ['en'], gpu=False, download_enabled=False, model_storage_directory='models', 
            detector=True, verbose=False
        )
        st.session_state.use_easyocr = True
    except Exception as e:
        st.warning(f"EasyOCR failed: {str(e)}. Switching to Tesseract OCR.")
        st.session_state.use_easyocr = False
        reader = None
    return reader

if st.session_state.ocr_reader is None:
    st.session_state.ocr_reader = load_ocr()

# --------- TEXT EXTRACTION FUNCTION ---------
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if st.session_state.use_easyocr and st.session_state.ocr_reader:
        try:
            return " ".join([text[1] for text in st.session_state.ocr_reader.readtext(gray)])
        except Exception as e:
            st.warning(f"EasyOCR failed: {str(e)}. Falling back to Tesseract.")

    try:
        return pytesseract.image_to_string(gray).strip()
    except Exception as e:
        st.error(f"Tesseract OCR failed: {str(e)}")
        return "OCR failed"

# --------- STREAMLIT UI ---------
st.title("TMKG Billboard Compliance Checker")
st.write("Upload an image of a billboard to check compliance.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Placeholder functions
def detect_tears(image):
    return image, "No", 0.0, "Stable", 1.0

def detect_obstruction(image):
    return image, "No", 0.0

def check_alignment(image):
    return 1.0

def analyze_brightness(image):
    return 1.0

# --------- IMAGE PROCESSING ---------
if uploaded_file is not None:
    try:
        if uploaded_file.size > 5_000_000:
            st.warning("⚠️ File is too large! Consider resizing.")
        else:
            MAX_SIZE = (1000, 1000)
            image = Image.open(uploaded_file)
            image.thumbnail(MAX_SIZE)

            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

            with st.spinner('🔍 Analyzing...'):
                tear_debug, torn, tear_conf, structure_status, structure_conf = detect_tears(image_cv)
                obstruct_mask, obstructed, obstruct_ratio = detect_obstruction(image_cv)
                align_conf = check_alignment(image_cv)
                brightness = analyze_brightness(image_cv)
                text_content = extract_text(image_cv)

            # --------- DISPLAY RESULTS ---------
            col1, col2 = st.columns(2)
            with col1:
                st.image(tear_debug, caption='Tear Analysis')
                st.image(obstruct_mask, caption=f'Obstruction: {obstructed} ({obstruct_ratio:.1%})')

            with col2:
                st.metric("Tear Status", f"{torn} ({tear_conf:.0%} Confidence)")
                st.metric("Structural Integrity", f"{structure_status} ({structure_conf:.0%} Confidence)")
                st.metric("Alignment", f"{align_conf:.0%} Confidence")
                st.metric("Brightness", f"{brightness:.0%} of Optimal")
                st.write("**Extracted Text:**", text_content if text_content else "No text found")

            # --------- COMPLIANCE SCORE ---------
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

            with st.expander("🔎 Penalty Breakdown"):
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
