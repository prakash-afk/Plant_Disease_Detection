import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
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
            border-radius: 14px;
            padding: 20px;
            margin-top: 10px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
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
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# TREATMENT TIPS
# --------------------------------------------------
TREATMENT_TIPS = {
    "Apple___Apple_scab": "Remove infected leaves, improve air circulation, and apply a recommended fungicide.",
    "Apple___Black_rot": "Prune infected branches and fruit, sanitize tools, and spray a suitable fungicide.",
    "Apple___Cedar_apple_rust": "Remove nearby alternate hosts if possible and apply preventive fungicide.",
    "Apple___healthy": "Plant looks healthy. Continue proper watering, sunlight, and regular monitoring.",

    "Blueberry___healthy": "Blueberry plant looks healthy. Maintain proper watering and nutrient balance.",

    "Cherry_(including_sour)___Powdery_mildew": "Improve airflow, avoid excess moisture, and apply sulfur or other suitable fungicide.",
    "Cherry_(including_sour)___healthy": "Cherry leaf looks healthy. Keep monitoring regularly.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Remove infected debris and use resistant varieties or fungicide if needed.",
    "Corn_(maize)___Common_rust_": "Use resistant hybrids and apply fungicide in severe infections.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use crop rotation and fungicide if infection is widespread.",
    "Corn_(maize)___healthy": "Corn plant looks healthy. Continue regular field monitoring.",

    "Grape___Black_rot": "Remove infected leaves and fruit, prune vines properly, and use fungicide.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood and maintain vineyard hygiene.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Improve ventilation and apply appropriate fungicide.",
    "Grape___healthy": "Grape leaf looks healthy. Maintain balanced irrigation and care.",

    "Orange___Haunglongbing_(Citrus_greening)": "Consult agricultural experts immediately. Remove infected plants where required and manage insect vectors.",
    
    "Peach___Bacterial_spot": "Avoid overhead watering, remove infected parts, and apply copper-based sprays if recommended.",
    "Peach___healthy": "Peach leaf looks healthy. Continue regular care.",

    "Pepper,_bell___Bacterial_spot": "Use disease-free seeds, remove infected leaves, and avoid water splash spread.",
    "Pepper,_bell___healthy": "Bell pepper plant looks healthy. Maintain proper spacing and watering.",

    "Potato___Early_blight": "Remove infected leaves and apply a recommended fungicide.",
    "Potato___Late_blight": "Destroy infected foliage quickly and avoid water splash spread.",
    "Potato___healthy": "Potato plant looks healthy. Keep checking for early disease symptoms.",

    "Raspberry___healthy": "Raspberry plant looks healthy. Continue normal maintenance.",

    "Soybean___healthy": "Soybean plant looks healthy. Maintain regular crop monitoring.",

    "Squash___Powdery_mildew": "Improve airflow, reduce leaf wetness, and apply suitable fungicide if needed.",

    "Strawberry___Leaf_scorch": "Remove infected leaves, avoid overhead watering, and improve air circulation.",
    "Strawberry___healthy": "Strawberry plant looks healthy. Keep monitoring for early symptoms.",

    "Tomato___Bacterial_spot": "Remove infected leaves, avoid overhead irrigation, and use copper-based treatment if appropriate.",
    "Tomato___Early_blight": "Remove affected leaves, avoid overhead watering, and apply fungicide if infection spreads.",
    "Tomato___Late_blight": "Remove infected plant parts immediately and apply a systemic fungicide.",
    "Tomato___Leaf_Mold": "Reduce humidity, improve ventilation, and use fungicide if needed.",
    "Tomato___Septoria_leaf_spot": "Remove lower infected leaves and apply protective fungicide if necessary.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray water under leaves, use miticide if severe, and improve plant health.",
    "Tomato___Target_Spot": "Remove infected leaves and use appropriate fungicide if required.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies, remove infected plants, and use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants, disinfect tools, and avoid tobacco contamination.",
    "Tomato___healthy": "Tomato plant looks healthy. Maintain consistent care and monitor regularly.",
}


def format_class_name(class_name: str) -> str:
    return class_name.replace("___", " - ").replace("_", " ")


def get_treatment_tip(class_name: str) -> str:
    return TREATMENT_TIPS.get(
        class_name,
        "Consult an agricultural expert for disease-specific treatment and prevention."
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 Plant Disease Detector")
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("1. Upload a clear leaf image")
    st.markdown("2. Click **Analyze Leaf**")
    st.markdown("3. View disease prediction, top 3 results, and treatment suggestion")
    st.markdown("---")
    st.markdown("**Best image tips**")
    st.markdown("- Use one leaf only")
    st.markdown("- Keep good lighting")
    st.markdown("- Avoid blurry photos")
    st.markdown("- Keep the leaf centered")
    st.markdown("---")
    st.markdown("**Backend**")
    st.caption("FastAPI prediction service")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image and get an instant AI-based disease prediction.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear, well-lit image for better prediction quality.",
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
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(API_URL, files=files, timeout=60)

                if response.status_code == 200:
                    result = response.json()

                    predicted_class = result.get("predicted_class", "Unknown")
                    confidence = float(result.get("confidence", 0))
                    top_predictions = result.get("top_predictions", [])

                    formatted_name = format_class_name(predicted_class)
                    is_healthy = "healthy" in predicted_class.lower()
                    card_class = "healthy-card" if is_healthy else "disease-card"
                    emoji = "✅" if is_healthy else "⚠️"

                    st.markdown("## 🩺 Diagnosis Result")

                    st.markdown(
                        f"""
                        <div class="result-card {card_class}">
                            <div class="diagnosis-title">{emoji} {formatted_name}</div>
                            <div class="meta-text" style="margin-top:8px;">
                                Confidence: <b>{confidence:.2f}%</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if confidence >= 80:
                        st.success("High-confidence prediction")
                    elif confidence >= 50:
                        st.warning("Medium-confidence prediction")
                    else:
                        st.error("Low-confidence prediction. Use a clearer image.")

                    st.markdown("### 💊 Treatment Suggestion")
                    st.info(get_treatment_tip(predicted_class))

                    st.markdown("### 📊 Top 3 Predictions")
                    for i, item in enumerate(top_predictions, start=1):
                        class_name = item["class_name"]
                        conf = float(item["confidence"])
                        label = format_class_name(class_name)

                        st.write(f"**{i}. {label}**")
                        st.progress(min(max(conf / 100, 0.0), 1.0))
                        st.caption(f"{conf:.2f}% confidence")

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
                st.error(
                    "Could not connect to FastAPI backend. Start it first using: python -m uvicorn app:app --reload"
                )
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