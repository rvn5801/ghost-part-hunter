import streamlit as st
import requests
from PIL import Image
import os
from pathlib import Path
from typing import Optional

# Configuration
API_URL = "http://localhost:8000"
BASE_DIR = Path(__file__).resolve().parent.parent.parent 

st.set_page_config(page_title="Ghost Part Hunter", layout="wide")

st.title("👻 Ghost Part Hunter")
st.markdown("### The Shazam for Broken Hardware")

# --- Initialize Session State (The App's Memory) ---
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

def smart_find_file(relative_path: str) -> Optional[str]:
    """
    Tries to find the file even if the folder structure changed.
    """
    if not relative_path:
        return None

    # 1. Try exact path
    full_path = BASE_DIR / relative_path
    if full_path.exists():
        return str(full_path)
    
    # 2. Try removing the category folder
    try:
        parts = Path(relative_path).parts
        if len(parts) > 3 and parts[1] == 'raw':
            new_parts = list(parts)
            new_parts.pop(2) # Remove 'gear'/'bracket' etc
            fallback_path = BASE_DIR / Path(*new_parts)
            if fallback_path.exists():
                return str(fallback_path)
    except Exception:
        pass
        
    return None

# --- Sidebar Controls ---
with st.sidebar:
    st.info("Instructions:")
    st.markdown("1. Place broken part on white paper.\n2. Place a **US Quarter** next to it.\n3. Snap a photo.")
    
    st.divider()
    # New Slider
    confidence_threshold = st.slider("Minimum Confidence", 0, 100, 75)

# --- File Upload Logic ---
uploaded_file = st.file_uploader("Upload a photo of the part", type=["jpg", "png", "jpeg"])
if not uploaded_file:
    uploaded_file = st.camera_input("Or take a picture")

# Check if user uploaded a NEW file, if so, clear old results
if uploaded_file:
    # We use size/name as a simple unique ID for the file
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.last_uploaded_file != file_id:
        st.session_state.search_results = None
        st.session_state.last_uploaded_file = file_id

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Part", width=300)
    
    # Button only triggers the API call
    if st.button("Find Replacement", type="primary"):
        with st.spinner("Analyzing Geometry & Texture..."):
            try:
                # Prepare file for API
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                
                response = requests.post(f"{API_URL}/search", files=files)
                
                if response.status_code == 200:
                    # SAVE results to session state
                    st.session_state.search_results = response.json()
                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Failed: {e}. Is the backend running?")

# --- Display Results (Persistent) ---
# This block runs even if you didn't just click "Find", provided we have results in memory
if st.session_state.search_results:
    results = st.session_state.search_results
    
    st.success(f"Found {len(results)} potential matches!")
    
    # Display Grid (Showing up to 20 results now)
    cols = st.columns(3)
    for idx, res in enumerate(results[:20]): 
        confidence = min(res['score'] * 100, 100.0)
        if confidence < confidence_threshold:  
            continue  
        with cols[idx % 3]:
            # Smart Path Finding
            real_thumb = smart_find_file(res['thumbnail_url'])
            real_stl = smart_find_file(res.get('stl_path', ''))

            st.markdown(f"**{idx+1}. {res['name']}**")
            
            if real_thumb:
                st.image(real_thumb, use_container_width=True)
            else:
                st.warning(f"Image missing")
            
            confidence = min(res['score'] * 100, 100.0)
            st.caption(f"Confidence: {confidence:.1f}%")
            
            # Dimensions logic could go here if coin detection was active
            st.caption(f"Size: {res['dimensions']['x']:.1f}mm wide")
            
            if real_stl:
                with open(real_stl, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download STL",
                        data=f,
                        file_name=os.path.basename(real_stl),
                        mime="model/stl",
                        key=f"dl_{idx}"
                    )
            else:
                st.error(f"STL File missing")
                
            st.divider()