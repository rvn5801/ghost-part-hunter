import streamlit as st
import requests
from PIL import Image
import os

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Ghost Part Hunter", layout="wide")

st.title("👻 Ghost Part Hunter")
st.markdown("### The Shazam for Broken Hardware")

# Sidebar for controls
with st.sidebar:
    st.info("Instructions:")
    st.markdown("1. Place broken part on white paper.\n2. Place a **US Quarter** next to it.\n3. Snap a photo.")

# Input Section
uploaded_file = st.file_uploader("Upload a photo of the part", type=["jpg", "png", "jpeg"])
if not uploaded_file:
    # Optional: Enable Camera
    uploaded_file = st.camera_input("Or take a picture")

if uploaded_file:
    # Display the User Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Part", width=300)
    
    if st.button("Find Replacement"):
        with st.spinner("Analyzing Geometry & Texture..."):
            try:
                # Prepare file for API
                # Reset pointer
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                
                # Call Backend
                response = requests.post(f"{API_URL}/search", files=files)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    st.success(f"Found {len(results)} potential matches!")
                    
                    # Display Results Grid
                    cols = st.columns(3)
                    for idx, res in enumerate(results[:6]): # Show top 6
                        with cols[idx % 3]:
                            # Construct local path for display
                            # Note: In production this would be a real URL
                            # We assume the frontend runs locally where data exists
                            thumb_path = res['thumbnail_url']
                            
                            st.markdown(f"**{res['name']}**")
                            
                            if os.path.exists(thumb_path):
                                st.image(thumb_path, use_container_width=True)
                            else:
                                st.warning("Image not found locally")
                            
                            st.caption(f"Confidence: {res['score']*100:.1f}%")
                            st.caption(f"Size: {res['dimensions']['x']:.1f}mm wide")
                            st.button(f"Download STL #{idx}", key=f"btn_{idx}")
                            st.divider()
                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Failed: {e}. Is the backend running?")