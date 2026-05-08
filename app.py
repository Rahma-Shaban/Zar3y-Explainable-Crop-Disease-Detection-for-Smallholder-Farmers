"""
Zar3y — Streamlit Demo (Requirement 5)

This is the main Streamlit application for the Zar3y crop disease
detection demo.

Features:
1. Upload a photo or use device camera
2. Send image to FastAPI backend (POST /predict)
3. Display prediction results:
   - Predicted disease class
   - Confidence score
   - Grad-CAM overlay (what the model looked at)
   - Plain-language description of the disease
   - Next-step suggestion (static dict, not LLM-generated)

Run with: streamlit run app.py
Opens at: http://localhost:8501
Backend must be running at: http://localhost:8000
"""

# TODO: Implement Streamlit frontend
#
# Key components to implement:
#
# import streamlit as st
# import requests
#
# st.set_page_config(page_title="Zar3y", page_icon="🌱", layout="wide")
#
# def main():
#     st.title("🌱 Zar3y — Crop Disease Detection")
#     st.markdown("Take a photo of a leaf to diagnose crop diseases.")
#
#     # Image input: upload or camera
#     # uploaded_file = st.file_uploader(...)
#     # camera_input = st.camera_input(...)
#
#     # Send to backend
#     # response = requests.post("http://localhost:8000/predict", ...)
#
#     # Display results:
#     # - Disease name + confidence
#     # - Grad-CAM overlay image
#     # - Plain-language description
#     # - Next-step recommendation
#
# if __name__ == "__main__":
#     main()
