# ============================================================
# TRAFFIC SIGN RECOGNITION SYSTEM
# Advanced Driver Assistance Systems (ADAS)
# Project Developed by: Kalyana Sundar - AI Engineer
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="🚦 Traffic Sign Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR ANIMATIONS & STYLING
# ============================================================

def load_css():
    """Load custom CSS for animations and professional styling"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto:wght@300;400;700&display=swap');
    
    /* Main Container Animation */
    .main-container {
        animation: fadeIn 1.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #0a0e27, #1a1a3e);
        padding: 30px;
        border-radius: 20px;
        border: 2px solid rgba(255, 107, 53, 0.3);
        box-shadow: 0 20px 60px rgba(255, 107, 53, 0.2);
        animation: headerGlow 3s ease-in-out infinite;
        margin-bottom: 30px;
        text-align: center;
    }
    
    @keyframes headerGlow {
        0%, 100% { box-shadow: 0 20px 60px rgba(255, 107, 53, 0.2); }
        50% { box-shadow: 0 20px 80px rgba(255, 107, 53, 0.4); }
    }
    
    .header-icon {
        font-size: 70px;
        animation: bounceIcon 2s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes bounceIcon {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .header-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 42px;
        background: linear-gradient(135deg, #ff6b35, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titlePulse 2s ease-in-out infinite;
        margin: 10px 0;
    }
    
    @keyframes titlePulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .header-subtitle {
        color: #8892b0;
        font-size: 18px;
        letter-spacing: 3px;
        font-weight: 300;
    }
    
    .developer-badge {
        background: rgba(255, 107, 53, 0.15);
        border: 1px solid rgba(255, 107, 53, 0.3);
        padding: 10px 30px;
        border-radius: 50px;
        display: inline-block;
        margin-top: 10px;
        color: #ffd93d;
        font-weight: 700;
        font-size: 16px;
        animation: fadeIn 2s ease-out;
    }
    
    .tech-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 12px;
        margin: 0 5px;
        font-weight: 600;
    }
    
    .badge-ml {
        background: rgba(255, 107, 53, 0.15);
        border: 1px solid rgba(255, 107, 53, 0.3);
        color: #ff6b35;
    }
    
    .badge-dl {
        background: rgba(78, 205, 196, 0.15);
        border: 1px solid rgba(78, 205, 196, 0.3);
        color: #4ecdc4;
    }
    
    .badge-adas {
        background: rgba(255, 217, 61, 0.15);
        border: 1px solid rgba(255, 217, 61, 0.3);
        color: #ffd93d;
    }
    
    /* Card Styling */
    .card-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 107, 53, 0.15);
        transition: all 0.5s ease;
        animation: slideUp 1s ease-out;
        margin-bottom: 20px;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(255, 107, 53, 0.1);
        border-color: rgba(255, 107, 53, 0.4);
    }
    
    .card-title {
        color: #ff6b35;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Result Container */
    .result-container {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(255, 217, 61, 0.05));
        border: 2px solid #ffd93d;
        border-radius: 15px;
        padding: 25px;
        animation: resultPulse 2s ease-in-out infinite;
        text-align: center;
    }
    
    @keyframes resultPulse {
        0%, 100% { border-color: #ffd93d; }
        50% { border-color: #ff6b35; }
    }
    
    .prediction-text {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #ffd93d, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 10px 0;
    }
    
    .confidence-text {
        font-size: 28px;
        font-weight: 700;
        color: #4ecdc4;
    }
    
    /* Custom Progress Bar */
    .custom-progress {
        background: #1a1a3e;
        border-radius: 10px;
        height: 25px;
        overflow: hidden;
        border: 1px solid rgba(255, 107, 53, 0.2);
        margin: 5px 0;
    }
    
    .custom-progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 1.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 11px;
        font-weight: 700;
    }
    
    /* Guide Steps */
    .guide-step {
        background: rgba(255, 255, 255, 0.03);
        padding: 12px 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #ff6b35;
        transition: all 0.3s ease;
    }
    
    .guide-step:hover {
        background: rgba(255, 107, 53, 0.08);
        transform: translateX(5px);
    }
    
    .step-number {
        color: #ff6b35;
        font-weight: 700;
        font-size: 14px;
        margin-right: 10px;
    }
    
    .step-text {
        color: #ccd6f6;
        font-size: 14px;
    }
    
    /* Sample Grid */
    .sample-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 15px 0;
    }
    
    .sample-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255, 107, 53, 0.1);
        transition: all 0.3s ease;
        cursor: default;
    }
    
    .sample-item:hover {
        transform: scale(1.05);
        border-color: #ff6b35;
        background: rgba(255, 107, 53, 0.1);
    }
    
    .sample-emoji {
        font-size: 35px;
        display: block;
        margin-bottom: 5px;
    }
    
    .sample-label {
        color: #8892b0;
        font-size: 11px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #495670;
        font-size: 13px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 30px;
    }
    
    .footer .highlight {
        color: #ff6b35;
        font-weight: 700;
    }
    
    .footer .developer {
        color: #ffd93d;
        font-weight: 700;
    }
    
    /* Responsive */
    @media (max-width: 600px) {
        .header-title { font-size: 28px; }
        .prediction-text { font-size: 24px; }
        .confidence-text { font-size: 20px; }
        .sample-grid { grid-template-columns: 1fr 1fr; }
        .header-icon { font-size: 50px; }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0e27;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff6b35, #ffd93d);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #ff6b35;
    }
    
    /* Spinner Animation */
    .stSpinner > div {
        border-color: #ff6b35 transparent #ffd93d transparent !important;
    }
    
    /* Success/Info Boxes */
    .stSuccess {
        background: rgba(78, 205, 196, 0.1) !important;
        border-color: #4ecdc4 !important;
    }
    
    .stInfo {
        background: rgba(255, 107, 53, 0.1) !important;
        border-color: #ff6b35 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND LABELS
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "TrafficSign_Best_Model.keras")
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_labels():
    """Load label mapping with caching"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        LABEL_PATH = os.path.join(BASE_DIR, "Label_Mapping.csv")
        return pd.read_csv(LABEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading labels: {str(e)}")
        return None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_class_emoji(class_name):
    """Get emoji for traffic sign class"""
    emoji_map = {
        "Stop": "🛑",
        "Speed": "🏁",
        "Turn": "🔄",
        "Left": "⬅️",
        "Right": "➡️",
        "Straight": "⬆️",
        "Roundabout": "🔄",
        "Crosswalk": "🚶",
        "Pedestrian": "🚶",
        "Bicycle": "🚲",
        "Bus": "🚌",
        "Car": "🚗",
        "Truck": "🚛",
        "No Entry": "🚫",
        "No Parking": "🅿️❌",
        "Parking": "🅿️",
        "Yield": "⏸️",
        "Warning": "⚠️",
        "Construction": "🚧",
        "Road Work": "🚧",
        "Railroad": "🚂",
        "School": "🏫",
        "Hospital": "🏥",
        "Gas Station": "⛽",
        "Restaurant": "🍽️",
        "Hotel": "🏨"
    }
    
    for key, emoji in emoji_map.items():
        if key.lower() in class_name.lower():
            return emoji
    return "🚦"

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    try:
        img = image.convert('RGB')
        img_array = np.array(img)
        img_resized = cv2.resize(img_array, (64, 64))
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def get_class_name(labels, class_id):
    """Get class name from labels DataFrame"""
    if "Label" in labels.columns:
        return labels.loc[labels["ClassId"] == class_id, "Label"].values[0]
    elif "SignName" in labels.columns:
        return labels.loc[labels["ClassId"] == class_id, "SignName"].values[0]
    else:
        return f"Class {class_id}"

# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application function"""
    
    # Load CSS
    load_css()
    
    # Load model and labels
    model = load_model()
    labels = load_labels()
    
    if model is None or labels is None:
        st.error("❌ Failed to load model or labels. Please check the files.")
        st.stop()
    
    # ============================================================
    # HEADER SECTION
    # ============================================================
    
    st.markdown("""
    <div class="header-container main-container">
        <div class="header-icon">🚦</div>
        <h1 class="header-title">Traffic Sign Recognition System</h1>
        <p class="header-subtitle">✦ ADVANCED DRIVER ASSISTANCE SYSTEMS (ADAS) ✦</p>
        <div class="developer-badge">
            👨‍💻 Developed by: Kalyana Sundar - AI Engineer
        </div>
        <div style="margin-top: 12px;">
            <span class="tech-badge badge-ml">🎯 Machine Learning</span>
            <span class="tech-badge badge-dl">🧠 Deep Learning</span>
            <span class="tech-badge badge-adas">🚗 ADAS</span>
            <span class="tech-badge" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #8892b0;">📊 GTSRB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # SIDEBAR - APP INFORMATION & GUIDE
    # ============================================================
    
    with st.sidebar:
        st.markdown("---")
        
        # What This App Does
        st.markdown("""
        <div class="card-container">
            <div class="card-title">🎯 What This App Does</div>
            <div style="color: #8892b0; font-size: 14px; line-height: 1.8;">
                <p>✅ <strong style="color: #ff6b35;">Recognizes</strong> traffic signs from uploaded images</p>
                <p>✅ <strong style="color: #ff6b35;">Classifies</strong> into 43 different traffic sign categories</p>
                <p>✅ <strong style="color: #ff6b35;">Provides</strong> confidence scores for predictions</p>
                <p>✅ <strong style="color: #ff6b35;">Shows</strong> top 5 most likely predictions</p>
                <p>✅ <strong style="color: #ff6b35;">Supports</strong> ADAS and autonomous vehicle systems</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # How to Use Guide
        st.markdown("""
        <div class="card-container">
            <div class="card-title">📖 How to Use</div>
            <div class="guide-step">
                <span class="step-number">Step 1:</span>
                <span class="step-text">Click "Browse files" below</span>
            </div>
            <div class="guide-step">
                <span class="step-number">Step 2:</span>
                <span class="step-text">Select a traffic sign image (PNG, JPG, JPEG)</span>
            </div>
            <div class="guide-step">
                <span class="step-number">Step 3:</span>
                <span class="step-text">Wait for AI to analyze the image</span>
            </div>
            <div class="guide-step">
                <span class="step-number">Step 4:</span>
                <span class="step-text">View the prediction with confidence score</span>
            </div>
            <div class="guide-step">
                <span class="step-number">Step 5:</span>
                <span class="step-text">Check top 5 predictions for comparison</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample Images
        st.markdown("""
        <div class="card-container">
            <div class="card-title">📸 Sample Signs</div>
            <div class="sample-grid">
                <div class="sample-item">
                    <span class="sample-emoji">🛑</span>
                    <span class="sample-label">Stop</span>
                </div>
                <div class="sample-item">
                    <span class="sample-emoji">⚠️</span>
                    <span class="sample-label">Warning</span>
                </div>
                <div class="sample-item">
                    <span class="sample-emoji">🚫</span>
                    <span class="sample-label">No Entry</span>
                </div>
                <div class="sample-item">
                    <span class="sample-emoji">🏁</span>
                    <span class="sample-label">Speed Limit</span>
                </div>
                <div class="sample-item">
                    <span class="sample-emoji">⬅️</span>
                    <span class="sample-label">Turn Left</span>
                </div>
                <div class="sample-item">
                    <span class="sample-emoji">➡️</span>
                    <span class="sample-label">Turn Right</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Performance
        st.markdown("""
        <div class="card-container">
            <div class="card-title">📊 Model Performance</div>
            <div style="color: #8892b0; font-size: 13px; line-height: 2;">
                <p><strong style="color: #ffd93d;">🏆 Accuracy:</strong> 99.86%</p>
                <p><strong style="color: #ffd93d;">📊 Dataset:</strong> GTSRB (43 classes)</p>
                <p><strong style="color: #ffd93d;">🧠 Architecture:</strong> Custom CNN</p>
                <p><strong style="color: #ffd93d;">📈 Training:</strong> 39,209 images</p>
                <p><strong style="color: #ffd93d;">🔄 Validation:</strong> 12,630 images</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Developer Info
        st.markdown("""
        <div class="card-container" style="border-color: #4ecdc4;">
            <div style="text-align: center;">
                <p style="color: #4ecdc4; font-size: 13px; margin: 0; font-weight: 600;">🚀 PROJECT DEVELOPED BY</p>
                <p style="color: #ffd93d; font-size: 18px; font-weight: 700; margin: 5px 0;">Kalyana Sundar</p>
                <p style="color: #8892b0; font-size: 13px; margin: 0;">AI Engineer</p>
                <p style="color: #495670; font-size: 11px; margin-top: 5px;">DataMites™ Project Mentoring</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # MAIN CONTENT - UPLOAD SECTION
    # ============================================================
    
    st.markdown("""
    <div class="card-container">
        <div class="card-title">📤 Upload Traffic Sign Image</div>
        <p style="color: #8892b0; font-size: 14px;">
            Upload a clear image of a traffic sign for instant recognition
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    # ============================================================
    # PROCESS UPLOADED IMAGE
    # ============================================================
    
    if uploaded is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="card-container">
                <div class="card-title" style="font-size: 16px;">🖼️ Uploaded Image</div>
            </div>
            """, unsafe_allow_html=True)
            
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Image info
            img_array = np.array(image)
            st.caption(f"📐 Dimensions: {img_array.shape[1]}×{img_array.shape[0]} px")
            st.caption(f"📁 Size: {uploaded.size/1024:.1f} KB")
        
        with col2:
            st.markdown("""
            <div class="card-container">
                <div class="card-title" style="font-size: 16px;">🎯 Prediction Result</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("🔄 Analyzing image with AI..."):
                # Preprocess
                processed_img = preprocess_image(image)
                
                if processed_img is not None:
                    # Predict
                    prediction = model.predict(processed_img, verbose=0)
                    
                    class_id = int(np.argmax(prediction))
                    confidence = float(np.max(prediction))
                    class_name = get_class_name(labels, class_id)
                    
                    # Display result with animation
                    emoji = get_class_emoji(class_name)
                    
                    st.markdown(f"""
                    <div class="result-container">
                        <div style="font-size: 60px; animation: bounceIcon 2s ease-in-out infinite;">{emoji}</div>
                        <div class="prediction-text">{class_name}</div>
                        <div style="margin: 10px 0;">
                            <span style="color: #8892b0; font-size: 16px;">Confidence</span>
                            <br>
                            <span class="confidence-text">{confidence*100:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence progress bar
                    st.markdown(f"""
                    <div style="margin: 15px 0;">
                        <div class="custom-progress">
                            <div class="custom-progress-bar" style="width: {confidence*100:.2f}%; background: linear-gradient(90deg, #ff6b35, #ffd93d);">
                                {confidence*100:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top 5 Predictions
                    st.markdown("""
                    <div style="margin-top: 20px;">
                        <h4 style="color: #8892b0;">📊 Top 5 Predictions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    top5 = np.argsort(prediction[0])[::-1][:5]
                    
                    for idx in top5:
                        name = get_class_name(labels, idx)
                        conf = prediction[0][idx]
                        emoji = get_class_emoji(name)
                        
                        # Color based on rank
                        bar_color = "linear-gradient(90deg, #ff6b35, #ffd93d)" if idx == top5[0] else "linear-gradient(90deg, #4ecdc4, #45b7d1)"
                        
                        st.markdown(f"""
                        <div style="margin: 8px 0;">
                            <div style="display: flex; justify-content: space-between; color: #ccd6f6; font-size: 13px; margin-bottom: 3px;">
                                <span>{emoji} {name}</span>
                                <span style="font-weight: 700; color: {'#ffd93d' if idx == top5[0] else '#4ecdc4'};">{conf*100:.2f}%</span>
                            </div>
                            <div class="custom-progress" style="height: 18px;">
                                <div class="custom-progress-bar" style="width: {conf*100:.2f}%; background: {bar_color}; font-size: 10px;">
                                    {conf*100:.1f}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # ============================================================
    # FOOTER
    # ============================================================
    
    st.markdown("""
    <div class="footer">
        <p style="margin: 0;">
            🚦 <span class="highlight">Traffic Sign Recognition System</span> &nbsp;|&nbsp;
            🎯 <span class="highlight">GTSRB Dataset</span> &nbsp;|&nbsp;
            🧠 <span class="highlight">Custom CNN Model</span>
        </p>
        <p style="margin: 5px 0; font-size: 12px;">
            👨‍💻 Developed by <span class="developer">Kalyana Sundar</span> - AI Engineer
        </p>
        <p style="margin: 5px 0; font-size: 11px; color: #495670;">
            DataMites™ Project Mentoring | © 2024 All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# RUN THE APP
# ============================================================

if __name__ == "__main__":
    main()