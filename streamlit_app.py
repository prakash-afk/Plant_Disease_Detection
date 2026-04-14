import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main {
        background-color: #f7f3ec;
    }

    .stButton > button {
        background-color: #2d6a35;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #24572b;
        color: white;
    }

    .result-card {
        background: white;
        border-radius: 14px;
        padding: 20px;
        margin-top: 10px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }

    .healthy-card {
        border-left: 6px solid #4caf50;
        background: #eaf7ea;
    }

    .disease-card {
        border-left: 6px solid #e67e22;
        background: #fff2e8;
    }

    .diagnosis-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2d1f;
    }

    .meta-text {
        color: #444;
        font-size: 0.98rem;
    }

    .small-note {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Plant Disease Detector")
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("1. Upload a clear leaf image")
    st.markdown("2. Click **Analyze Leaf**")
    st.markdown("3. View disease prediction and confidence")
    st.markdown("---")
    st.markdown("**Best image tips**")
    st.markdown("- Use one leaf only")
    st.markdown("- Keep good lighting")
    st.markdown("- Avoid blurry photos")
    st.markdown("- Keep the leaf centered")
    st.markdown("---")
    st.markdown("**Backend**")
    st.caption("FastAPI prediction service")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image and get an instant AI-based disease prediction.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear, well-lit image for better prediction quality."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1.05, 1])

    with col1:
        st.image(image, caption="Uploaded leaf image", use_container_width=True)

    with col2:
        st.markdown("**Image Details**")
        st.write(f"📐 Size: {image.size[0]} × {image.size[1]} px")
        st.write(f"🎨 Mode: {image.mode}")
        st.write(f"📁 File: {uploaded_file.name}")
        st.write("")

        analyze = st.button("🔍 Analyze Leaf")

    if analyze:
        with st.spinner("Analyzing image..."):
            try:
                uploaded_file.seek(0)
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                response = requests.post(API_URL, files=files, timeout=60)

                if response.status_code == 200:
                    result = response.json()

                    predicted_class = result.get("predicted_class", "Unknown")
                    confidence = float(result.get("confidence", 0))

                    is_healthy = "healthy" in predicted_class.lower()
                    card_class = "healthy-card" if is_healthy else "disease-card"
                    emoji = "✅" if is_healthy else "⚠️"

                    st.markdown("## 🩺 Diagnosis Result")

                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div class="diagnosis-title">{emoji} {predicted_class.replace('_', ' ')}</div>
                        <div class="meta-text" style="margin-top:8px;">
                            Confidence: <b>{confidence:.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if confidence >= 80:
                        st.success("High-confidence prediction")
                    elif confidence >= 50:
                        st.warning("Medium-confidence prediction")
                    else:
                        st.error("Low-confidence prediction. Use a clearer image.")

                    st.markdown("### 📌 What this means")
                    if is_healthy:
                        st.info("The model thinks this leaf looks healthy.")
                    else:
                        st.info("The model detected a disease class from the trained dataset.")

                    st.markdown("---")
                    st.caption(
                        "This tool is for educational and project purposes. "
                        "For real agricultural decisions, consult an agronomist or plant expert."
                    )

                else:
                    st.error(f"API error: {response.status_code}")
                    try:
                        st.json(response.json())
                    except Exception:
                        st.write(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Start the backend first with: uvicorn app:app --reload")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try again with a smaller or clearer image.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

else:
    st.markdown("### 👆 Upload a leaf photo to begin")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("📸 **Clear image**  \nUse a sharp leaf photo")
    with c2:
        st.markdown("🎯 **Single subject**  \nFocus on one leaf")
    with c3:
        st.markdown("⚡ **Instant result**  \nPrediction in seconds")