import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os
from dotenv import load_dotenv
import tempfile
import json
import cv2
import firebase_admin   
from firebase_admin import credentials, firestore
from datetime import datetime
import google.generativeai as genai

GEMINI_API_KEY = st.secrets["gemini"]["api_key"]

# Debug toggle
DEBUG = False

# Load class names from JSON file
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load environment variables before dep/test in offline
#load_dotenv()
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

# Configure Gemini and ElevenLabs
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Firebase only if not already initialized
if not firebase_admin._apps:
    # Convert AttrDict to regular dict
    firebase_json = dict(st.secrets["firebase"])

    # Create a temporary JSON file to store credentials
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp_file:
        json.dump(firebase_json, tmp_file)
        tmp_path = tmp_file.name

    # Initialize Firebase
    cred = credentials.Certificate(tmp_path)
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# Load Keras model (updated line)
model = tf.keras.models.load_model("waste_classifier_model.h5", compile=False)

def extract_roi(image: Image.Image) -> np.ndarray:
    img = image.copy().convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        roi = img_cv[y:y + h, x:x + w]
        if w < 10 or h < 10:
            roi = img_cv
        else:
            roi = cv2.resize(roi, (224, 224))
    else:
        roi = cv2.resize(img_cv, (224, 224))

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    if DEBUG:
        st.image(roi_rgb, caption="ROI", use_column_width=True)
    return roi_rgb

def classify_image_keras(img: Image.Image) -> tuple:
    roi = extract_roi(img)
    img_array = np.array(roi).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100
    return class_names[predicted_class].replace("_", " "), confidence

def get_disposal_advice(item: str, feedback: str = "") -> str:
    prompt = f"""
You are GreenGaze — an eco-conscious AI aligned with India's Sustainable Development Goal 11.

Your task:
Generate **friendly yet professional** disposal advice for the item: "{item}".

Output should include:
1. **Step-by-step disposal instructions** (broken down clearly)
2. A **DIY reuse idea** if the item can be reused (clearly step-by-step like a Pinterest hack
3. A quick **tip or link** to help locate disposal spots
4. Keep it **concise, clear**, and **India-focused**
5.Don.t mention **SDG 11**
6. Use **emojis**
7. Use **user friendly text**

{f"User feedback: {feedback}" if feedback else ""}

Respond in Markdown format with icons if useful.
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error from Gemini API: {str(e)}"

def speak_text(text: str):
    st.info("🔊 Voice instructions coming soon!")
    return None

def save_metadata_to_firestore(label, confidence, feedback=""):
    now = datetime.now().isoformat()

    # Create a reference to the collection for the class
    collection_name = "waste_predictions"

    # Create a reference to the document (document ID is auto-generated)
    # Store each class as a subcollection within the main "waste_predictions" collection
    doc_ref = db.collection(collection_name).document(label).collection("records").document()  # auto-generate document ID

    # Set the document data
    doc_ref.set({
        "label": label,
        "confidence": confidence,
        "feedback": feedback,
        "timestamp": now
    })

# UI Setup

# Sidebar Navigation
st.sidebar.title("🌿 GreenGaze")
selected = st.sidebar.radio("Go to", ["Home", "About", "Privacy Policy", "Contact"])
if selected == "Home":
 st.markdown("""
    <style>
        .main {
            font-family: 'Segoe UI', sans-serif;
            color: #1B5E20;
        }
        h1, h2 {
            text-align: center;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
        }
        .stRadio > div {
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

 st.markdown("<h1 style='color:#1B5E20;'>🌿 GreenGaze</h1>", unsafe_allow_html=True)
 st.caption("Snap. Detect. Dispose smart.SDG11 ♻️")

 # Input
 with st.expander("📸 Choose Input Method", expanded=True):
    input_method = st.radio("", ("Take a Photo", "Upload an Image"), horizontal=True)
    image_obj = None

    if input_method == "Take a Photo":
        cam_img = st.camera_input("Capture image")
        if cam_img:
            image_obj = Image.open(cam_img)

    elif input_method == "Upload an Image":
        uploaded_img = st.file_uploader("Upload a waste item image", type=["jpg", "jpeg", "png"])
        if uploaded_img:
            image_obj = Image.open(uploaded_img)

 # Process & Results
 if image_obj:
    st.image(image_obj, caption="📷 Your Image", use_container_width=True)

    with st.spinner("🔍 Classifying image..."):
        item, confidence = classify_image_keras(image_obj)

    st.success(f"✅ Identified as: **{item}** with **{confidence:.2f}%** confidence")

    with st.spinner("🧠 Generating smart disposal advice..."):
        advice = get_disposal_advice(item)
    with st.expander("🗑️ Disposal & DIY Guide", expanded=True):
        st.markdown(advice, unsafe_allow_html=True)

    audio_path = speak_text(advice)
    if audio_path:
        with open(audio_path, 'rb') as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")
        os.remove(audio_path)

    # Feedback
    with st.expander("✍️ Feedback or Correction"):
        feedback = st.text_input("Think the item is different? Suggest a correction:")
        if feedback:
            st.markdown("🔄 Updating advice based on your suggestion...")
            updated_advice = get_disposal_advice(item, feedback)
            st.warning("🔁 Updated Advice:")
            st.markdown(updated_advice)

            audio_path = speak_text(updated_advice)
            if audio_path:
                with open(audio_path, 'rb') as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")
                os.remove(audio_path)

   # 🌟 Contribute Section 
 with st.expander("📊 Help Us Improve!", expanded=True):
    st.markdown("""
        <div style='background-color:#E8F5E9; padding:20px; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.1); margin-bottom:15px;'>
            <h4 style='color:#1B5E20; margin-bottom:10px;'>🌿 Contribute to Sustainability</h4>
            <p style='color:#1B5E20; font-size:15px; margin-bottom:0;'>By sharing your result, you're helping us make GreenGaze smarter. Data is anonymized and used only for improvement. 💚</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        consent = st.checkbox("Yes", value=True, key="save_consent")
    with col2:
        st.markdown("**I consent to share this result for model training**")

    submit_button = st.button("📤 Submit to Firestore")

    if submit_button and consent:
        save_metadata_to_firestore(item, confidence, feedback)
        st.success("✅ Thank you! Your result has been saved.")
    elif submit_button and not consent:
        st.error("⚠️ Please check the box to give your consent.")

    st.markdown("<hr style='border: 1px solid #ddd;'/>", unsafe_allow_html=True)

# ℹ️ About Page
elif selected == "About":
    st.markdown("<h1 style='color:#2E7D32;'>🌿 About GreenGaze</h1>", unsafe_allow_html=True)

    st.image("https://cdn-icons-png.flaticon.com/512/4299/4299917.png", width=100)  # Optional: Green icon

    st.markdown("""
    **GreenGaze** is an eco-conscious AI tool designed to make waste disposal smarter, faster, and greener. ♻️  
    It classifies waste through image analysis and gives **intelligent, sustainable suggestions** using Google Gemini.

    ---
    ### 🌏 Why GreenGaze?
    - ✅ Aligned with **India’s Sustainable Development Goals** 🇮🇳  
    - 🧠 Powered by **AI + Google Gemini** for smart waste advice  
    - ❤️ Built with **Streamlit**, simple and fast  
    - 📸 Just **upload an image**, and we handle the rest  

    ---
    ### 🌱 Our Mission
    > "To make sustainability easy, accessible, and part of everyday choices."

    ---
    ### 👥 Who is it for?
    - Environment enthusiasts  
    - Schools & colleges  
    - Municipal bodies  
    - NGOs promoting zero-waste goals  
    """)

# 🔒 Privacy Policy
elif selected == "Privacy Policy":
    st.markdown("<h1 style='color:#2E7D32;'>🔒 Privacy Policy</h1>", unsafe_allow_html=True)

    st.markdown("""
    We take your privacy seriously. 🌱 Here's what you need to know:

    ---
    ### ✅ What We **Don’t** Do:
    - ❌ We **don’t collect** personal data like name, address, or phone number  
    - ❌ We **don’t track** your activity across the web  
    - ❌ We **never sell** or share data with third parties

    ---
    ### 📊 What We **Do** Collect:
    - ✅ Uploaded images (temporarily, for AI classification)  
    - ✅ Feedback you provide (voluntarily)  
    - ✅ Anonymized usage data (to improve performance)

    ---
    ### 🔐 Data Handling
    - Data is stored **securely & temporarily**
    - All data is **anonymized**
    - You can **opt out** anytime

    ---
    For any privacy concerns, contact us:  
    📧 `support@greengaze.app`
    """)

# 📬 Contact Page
elif selected == "Contact":
    st.markdown("<h1 style='color:#2E7D32;'>📬 Contact Us</h1>", unsafe_allow_html=True)

    st
    