import io
import requests
import streamlit as st

from PIL import Image
import base64


# --------------------------
# Config
# --------------------------
BACKEND_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{BACKEND_URL}/predict"


st.set_page_config(
    page_title="Zar3y",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Zar3y")
st.subheader("AI Crop Disease Detection")


# --------------------------
# Backend health check
# --------------------------
def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False


backend_ok = check_backend()

if not backend_ok:
    st.error("⚠ Backend is not running! Please start FastAPI first.")
    st.stop()


# --------------------------
# Upload image
# --------------------------
uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):

        files = {
            "file": (
                "leaf.jpg",
                uploaded_file.getvalue(),
                "image/jpeg"
            )
        }

        try:
            response = requests.post(
                PREDICT_URL,
                files=files,
                timeout=20
            )

            response.raise_for_status()
            result = response.json()

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()


    # --------------------------
    # Results
    # --------------------------
    with col2:

        st.success(result["display_name"])

        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )

        st.write("### Description")
        st.write(result["description"])

        st.write("### Recommended Action")
        st.write(result["next_step"])

        st.write("### Inference Time")
        st.write(f"{result['inference_ms']} ms")


    # --------------------------
    # Grad-CAM
    # --------------------------
    gradcam_bytes = base64.b64decode(result["grad_cam_base64"])
    gradcam_image = Image.open(io.BytesIO(gradcam_bytes))

    st.write("## Grad-CAM")

    g1, g2 = st.columns(2)

    with g1:
        st.image(image, caption="Original")

    with g2:
        st.image(gradcam_image, caption="Heatmap")
