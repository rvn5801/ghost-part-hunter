import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import cv2

# Import our new Dynamic Detector
try:
    from src.core.vision.coin_detector import CoinDetector
    vision_system = CoinDetector()
except ImportError:
    vision_system = None

try:
    from config.settings import settings
except ImportError:
    pass

# --- Configuration ---
API_URL = "http://localhost:8000"
BASE_DIR = PROJECT_ROOT 

st.set_page_config(page_title="Ghost Part Hunter v2.0", layout="wide")

# --- CSS FOR STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #FF4B4B;}
    .sub-text {font-size: 1.1rem; color: #555;}
    .match-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar: The Engineer's Toolkit ---
with st.sidebar:
    st.title("⚙️ Calibration")
    
    # 1. Multi-Currency Support
    st.markdown("### 1. Select Reference Object")
    ref_options = {
        "US Quarter": 24.26,
        "US Penny": 19.05,
        "Euro (€1)": 23.25,
        "Credit Card (Width)": 85.60,
        "Custom (mm)": 0.0
    }
    selected_ref = st.selectbox("What is next to the part?", list(ref_options.keys()))
    
    ref_size = ref_options[selected_ref]
    if selected_ref == "Custom (mm)":
        ref_size = st.number_input("Enter diameter/width in mm", value=25.0)

    st.divider()

    # 2. AI Tools
    st.markdown("### 2. Vision Tools")
    use_ai_bg_removal = st.toggle("AI Background Removal", value=False, help="Turn this ON if you have a messy desk/background.")

    st.divider()
    
    # 3. Search Sensitivity
    st.markdown("### 3. Search Sensitivity")
    threshold = st.slider("Confidence Threshold", 0, 100, 70, help="Only show matches above this score.")
    
    st.divider()
    st.info("💡 **Tip:** Ensure good lighting. Shadows can look like part geometry.")

# --- Main App ---
st.markdown('<p class="main-header">👻 Ghost Part Hunter</p>', unsafe_allow_html=True)
st.caption(f"Calibrated for: **{selected_ref}** ({ref_size}mm)")

# --- Session State ---
if 'search_results' not in st.session_state: st.session_state.search_results = None
if 'processed_image' not in st.session_state: st.session_state.processed_image = None

# --- Upload ---
uploaded_file = st.file_uploader("Upload Part Photo", type=["jpg", "png", "jpeg"])
if not uploaded_file:
    uploaded_file = st.camera_input("Or take a picture")

if uploaded_file:
    # 1. Read the file
    file_bytes = uploaded_file.getvalue()
    
    # 2. AI Background Removal (Optional Step)
    if use_ai_bg_removal and vision_system:
        with st.spinner("✨ AI is removing the background..."):
            clean_cv_img = vision_system.remove_background(file_bytes)
            # Encode for backend (OpenCV uses BGR, which is fine for JPEG encoding)
            is_success, buffer = cv2.imencode(".jpg", clean_cv_img)
            if is_success:
                file_bytes = buffer.tobytes()
                st.session_state.processed_image = clean_cv_img
    
    # Display the Image
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.session_state.processed_image is not None and use_ai_bg_removal:
            # FIX: Convert BGR (OpenCV) to RGB (Streamlit) for correct display
            rgb_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption="AI Processed View (Computer Vision)", use_container_width=True)
        else:
            st.image(uploaded_file, caption="Original Upload", use_container_width=True)

    # 3. Action Button
    if st.button("🔍 Analyze & Find Part", type="primary", use_container_width=True):
        with st.spinner("⚡ Measuring & Searching..."):
            files = {"file": ("processed.jpg", file_bytes, "image/jpeg")}
            
            try:
                response = requests.post(f"{API_URL}/search", files=files)
                if response.status_code == 200:
                    st.session_state.search_results = response.json()
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# --- Results Display ---
if st.session_state.search_results:
    # FILTER RESULTS BASED ON THRESHOLD
    all_results = st.session_state.search_results
    results = [r for r in all_results if (r['score'] * 100) >= threshold]
    
    if not results:
        st.warning(f"No matches found above {threshold}% confidence.")
    else:
        st.success(f"✅ Found {len(results)} matches above {threshold}% confidence!")

        # --- 🏆 FEATURED RESULT (Top Match) ---
        top_match = results[0]
        
        top_thumb_path = None
        if top_match['thumbnail_url']:
            p = BASE_DIR / top_match['thumbnail_url']
            if p.exists(): top_thumb_path = str(p)

        st.markdown("### 🥇 Best Match & AI Verification")
        
        with st.container(border=True):
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                if top_thumb_path:
                    st.image(top_thumb_path, use_container_width=True)
                else:
                    st.write("No Image Available")
            
            with col2:
                st.header(f"1. {top_match['name']}")
                confidence = min(top_match['score'] * 100, 100.0)
                st.progress(confidence / 100, text=f"Confidence: {confidence:.1f}%")
                
                # Gemini Analysis Box
                if top_match.get('gemini_analysis') and top_match['gemini_analysis'] != "Pending":
                    analysis = top_match['gemini_analysis']
                    if "Gemini not configured" in analysis:
                        st.warning("⚠️ Gemini verification skipped (Not Configured)")
                    else:
                        st.info(f"🤖 **Gemini AI Analysis:**\n\n{analysis}")
                else:
                    st.caption("No AI analysis available.")

        # --- 🏁 GRID FOR REMAINING MATCHES ---
        if len(results) > 1:
            st.markdown("---")
            st.subheader(f"🔍 Other Candidates ({len(results)-1})")
            
            other_matches = results[1:] 
            
            grid_cols = st.columns(3)
            
            for i, match in enumerate(other_matches):
                with grid_cols[i % 3]: 
                    with st.container(border=True):
                        thumb_path = None
                        if match['thumbnail_url']:
                            p = BASE_DIR / match['thumbnail_url']
                            if p.exists(): thumb_path = str(p)

                        if thumb_path:
                            st.image(thumb_path, use_container_width=True)
                        
                        st.markdown(f"**{i+2}. {match['name']}**")
                        
                        score_val = min(match['score'] * 100, 100.0)
                        st.caption(f"Score: {score_val:.1f}%")