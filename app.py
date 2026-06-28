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
import base64
from streamlit.components.v1 import html

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
    """Load custom CSS for animations and styling"""
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
    
    /* Header Animation */
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
    
    .header-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 42px;
        background: linear-gradient(135deg, #ff6b35, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titlePulse 2s ease-in-out infinite;
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
        padding: 10px 25px;
        border-radius: 50px;
        display: inline-block;
        margin-top: 10px;
        color: #ffd93d;
        font-weight: 700;
        animation: fadeIn 2s ease-out;
    }
    
    /* Card Animations */
    .card-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 107, 53, 0.2);
        transition: all 0.5s ease;
        animation: slideUp 1s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(255, 107, 53, 0.15);
        border-color: #ff6b35;
    }
    
    /* Prediction Result Animation */
    .result-container {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(255, 217, 61, 0.05));
        border: 2px solid #ffd93d;
        border-radius: 15px;
        padding: 20px;
        animation: resultPulse 2s ease-in-out infinite;
    }
    
    @keyframes resultPulse {
        0%, 100% { border-color: #ffd93d; }
        50% { border-color: #ff6b35; }
    }
    
    .prediction-text {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #ffd93d, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .confidence-text {
        font-size: 24px;
        font-weight: 700;
        color: #4ecdc4;
    }
    
    /* Progress Bar Animation */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff6b35, #ffd93d) !important;
        animation: progressGlow 2s ease-in-out infinite;
    }
    
    @keyframes progressGlow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Sidebar Styling */
    .sidebar-container {
        background: linear-gradient(180deg, #0a0e27, #1a1a3e);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 107, 53, 0.2);
        margin-bottom: 20px;
    }
    
    .sidebar-title {
        color: #ff6b35;
        font-weight: 700;
        font-size: 18px;
        border-bottom: 2px solid rgba(255, 107, 53, 0.3);
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    
    /* Guide Section */
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
    
    .guide-step .step-number {
        color: #ff6b35;
        font-weight: 700;
        font-size: 14px;
    }
    
    .guide-step .step-text {
        color: #ccd6f6;
        font-size: 14px;
    }
    
    /* Sample Images Grid */
    .sample-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin: 15px 0;
    }
    
    .sample-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255, 107, 53, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .sample-item:hover {
        transform: scale(1.05);
        border-color: #ff6b35;
        box-shadow: 0 5px 20px rgba(255, 107, 53, 0.2);
    }
    
    .sample-item .sample-label {
        color: #ccd6f6;
        font-size: 12px;
        margin-top: 5px;
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
    
    /* Responsive Design */
    @media (max-width: 600px) {
        .header-title { font-size: 28px; }
        .prediction-text { font-size: 24px; }
        .confidence-text { font-size: 18px; }
        .sample-grid { grid-template-columns: 1fr 1fr; }
    }
    
    /* Scrollbar Styling */
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
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND LABELS
# ============================================================

@st.cache_resource
def load_model_and_labels():
    """Load the trained model and label mapping with caching"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "CNN_Final.keras")
        LABEL_PATH = os.path.join(BASE_DIR, "Label_Mapping.csv")
        
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        labels = pd.read_csv(LABEL_PATH)
        return model, labels
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# ============================================================
# PREPROCESS IMAGE FUNCTION
# ============================================================

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    try:
        # Convert to RGB if needed
        img = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Resize to 64x64
        img_resized = cv2.resize(img_array, (64, 64))
        
        # Normalize
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

# ============================================================
# DISPLAY SAMPLE IMAGES FUNCTION
# ============================================================

def display_sample_images():
    """Display sample traffic sign images with their labels"""
    st.markdown("""
    <div class="card-container">
        <h4 style="color: #ff6b35; margin-bottom: 15px;">📸 Sample Traffic Signs</h4>
        <div class="sample-grid">
            <div class="sample-item">
                <span style="font-size: 40px;">🚦</span>
                <div class="sample-label">Stop Sign</div>
            </div>
            <div class="sample-item">
                <span style="font-size: 40px;">⚠️</span>
                <div class="sample-label">Warning</div>
            </div>
            <div class="sample-item">
                <span style="font-size: 40px;">⬅️</span>
                <div class="sample-label">Turn Left</div>
            </div>
            <div class="sample-item">
                <span style="font-size: 40px;">➡️</span>
                <div class="sample-label">Turn Right</div>
            </div>
            <div class="sample-item">
                <span style="font-size: 40px;">🚫</span>
                <div class="sample-label">No Entry</div>
            </div>
            <div class="sample-item">
                <span style="font-size: 40px;">🏁</span>
                <div class="sample-label">Speed Limit</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application function"""
    
    # Load CSS
    load_css()
    
    # ============================================================
    # HEADER SECTION
    # ============================================================
    
    st.markdown("""
    <div class="header-container main-container">
        <div style="font-size: 60px; animation: bounceIcon 2s ease-in-out infinite;">🚦</div>
        <h1 class="header-title">Traffic Sign Recognition System</h1>
        <p class="header-subtitle">✦ ADVANCED DRIVER ASSISTANCE SYSTEMS (ADAS) ✦</p>
        <div class="developer-badge">
            👨‍💻 Developed by: Kalyana Sundar - AI Engineer
        </div>
        <div style="margin-top: 10px;">
            <span class="badge" style="background: rgba(255,107,53,0.15); border: 1px solid rgba(255,107,53,0.3); padding: 5px 15px; border-radius: 50px; color: #ff6b35; font-size: 12px; margin: 0 5px;">🎯 Machine Learning</span>
            <span class="badge" style="background: rgba(78,205,196,0.15); border: 1px solid rgba(78,205,196,0.3); padding: 5px 15px; border-radius: 50px; color: #4ecdc4; font-size: 12px; margin: 0 5px;">🧠 Deep Learning</span>
            <span class="badge" style="background: rgba(255,217,61,0.15); border: 1px solid rgba(255,217,61,0.3); padding: 5px 15px; border-radius: 50px; color: #ffd93d; font-size: 12px; margin: 0 5px;">🚗 ADAS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # SIDEBAR - APP INFORMATION & GUIDE
    # ============================================================
    
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-container">
            <div class="sidebar-title">ℹ️ About This App</div>
            <p style="color: #8892b0; font-size: 13px; line-height: 1.6;">
                This application uses <strong style="color: #ff6b35;">Deep Learning</strong> to automatically 
                recognize and classify traffic signs from uploaded images. It's designed to assist 
                <strong style="color: #ffd93d;">Advanced Driver Assistance Systems (ADAS)</strong> 
                and <strong style="color: #ffd93d;">Autonomous Vehicles</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-container">
            <div class="sidebar-title">🎯 What This App Does</div>
            <ul style="color: #8892b0; font-size: 13px; line-height: 1.8; padding-left: 20px;">
                <li>📸 Recognizes traffic signs from images</li>
                <li>🎯 Classifies into 43 categories</li>
                <li>📊 Provides confidence scores</li>
                <li>⚡ Real-time predictions</li>
                <li>🚗 Supports ADAS systems</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-container">
            <div class="sidebar-title">📖 How to Use This App</div>
            <div class="guide-step">
                <span class="step-number">Step 1:</span>
                <span class="step-text">Upload a traffic sign image (PNG, JPG, JPEG)</span>
            </div>
            <div class="guide-step">
                <span class="step-number">Step 2:</span>
                <span class="step-text">Wait for the model to process the image</span>
            </div>
            <div class="guide-step">
                <span class="step-number">Step 3:</span>
                <span class="step-text">View the predicted sign and confidence</span>
            </div>
            <div class="guide-step">
                <span class="step-number">Step 4:</span>
                <span class="step-text">Use the result for your application</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-container">
            <div class="sidebar-title">📊 Model Performance</div>
            <div style="color: #8892b0; font-size: 13px; line-height: 1.8;">
                <p><strong style="color: #ffd93d;">🏆 Accuracy:</strong> 99.86%</p>
                <p><strong style="color: #ffd93d;">📊 Dataset:</strong> GTSRB (43 classes)</p>
                <p><strong style="color: #ffd93d;">🧠 Architecture:</strong> Custom CNN</p>
                <p><strong style="color: #ffd93d;">📈 Training Images:</strong> 39,209</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample Images
        display_sample_images()
        
        # Developer Info
        st.markdown("""
        <div class="sidebar-container" style="border-color: #4ecdc4;">
            <div style="text-align: center;">
                <p style="color: #4ecdc4; font-size: 12px; margin: 0;">
                    🚀 Project Developed by<br>
                    <strong style="color: #ffd93d; font-size: 14px;">Kalyana Sundar</strong><br>
                    <span style="color: #8892b0; font-size: 11px;">AI Engineer</span>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # MAIN CONTENT AREA
    # ============================================================
    
    # Load model and labels
    model, labels = load_model_and_labels()
    
    if model is None or labels is None:
        st.error("❌ Failed to load model. Please check the model files.")
        return
    
    # ============================================================
    # FILE UPLOADER
    # ============================================================
    
    st.markdown("""
    <div class="card-container">
        <h3 style="color: #ff6b35;">📤 Upload Traffic Sign Image</h3>
        <p style="color: #8892b0; font-size: 14px;">
            Upload an image of a traffic sign for instant recognition
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of a traffic sign for best results"
    )
    
    # ============================================================
    # PROCESS UPLOADED IMAGE
    # ============================================================
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="card-container">
                <h4 style="color: #8892b0;">🖼️ Uploaded Image</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Display image info
            img_array = np.array(image)
            st.caption(f"📐 Image Size: {img_array.shape[1]}×{img_array.shape[0]} pixels")
            st.caption(f"📁 File: {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
        
        with col2:
            st.markdown("""
            <div class="card-container">
                <h4 style="color: #8892b0;">🎯 Prediction Result</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Show progress
            with st.spinner("🔄 Analyzing image..."):
                # Preprocess image
                processed_img = preprocess_image(image)
                
                if processed_img is not None:
                    # Make prediction
                    prediction = model.predict(processed_img)
                    class_id = int(np.argmax(prediction))
                    confidence = float(np.max(prediction))
                    
                    # Get class name
                    class_name = labels.loc[
                        labels["ClassId"] == class_id,
                        "Label"
                    ].values[0]
                    
                    # Display results with animation
                    st.markdown(f"""
                    <div class="result-container">
                        <div style="text-align: center;">
                            <div style="font-size: 48px; margin-bottom: 10px;">{get_class_emoji(class_name)}</div>
                            <div class="prediction-text">{class_name}</div>
                            <div style="margin: 10px 0;">
                                <span style="color: #8892b0;">Confidence:</span>
                                <span class="confidence-text">{confidence*100:.2f}%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence progress bar
                    st.markdown(f"""
                    <div style="margin: 15px 0;">
                        <div style="background: #1a1a3e; border-radius: 10px; height: 20px; overflow: hidden; border: 1px solid rgba(255,107,53,0.3);">
                            <div style="width: {confidence*100:.2f}%; background: linear-gradient(90deg, #ff6b35, #ffd93d); height: 100%; border-radius: 10px; transition: width 1s ease; display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: 700;">
                                {confidence*100:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top predictions
                    st.markdown("""
                    <h5 style="color: #8892b0; margin-top: 15px;">📊 Top Predictions</h5>
                    """, unsafe_allow_html=True)
                    
                    top_indices = np.argsort(prediction[0])[-5:][::-1]
                    for i, idx in enumerate(top_indices):
                        conf = prediction[0][idx]
                        name = labels.loc[labels["ClassId"] == idx, "Label"].values[0]
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <span style="color: #8892b0; width: 80px; font-size: 12px;">{i+1}. {name}</span>
                            <div style="flex: 1; background: #1a1a3e; border-radius: 5px; height: 15px; overflow: hidden; margin: 0 10px;">
                                <div style="width: {conf*100:.1f}%; background: {'linear-gradient(90deg, #ff6b35, #ffd93d)' if i==0 else 'linear-gradient(90deg, #4ecdc4, #45b7d1)'}; height: 100%; border-radius: 5px; transition: width 1s ease;"></div>
                            </div>
                            <span style="color: #ccd6f6; font-size: 12px; width: 60px;">{conf*100:.1f}%</span>
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
            👨‍💻 Developed by <strong style="color: #ffd93d;">Kalyana Sundar</strong> - AI Engineer
        </p>
        <p style="margin: 5px 0; font-size: 11px; color: #495670;">
            © 2024 All Rights Reserved | Advanced Driver Assistance Systems (ADAS)
        </p>
    </div>
    """, unsafe_allow_html=True)

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
    
    # Return matching emoji or default
    for key, emoji in emoji_map.items():
        if key.lower() in class_name.lower():
            return emoji
    return "🚦"

# ============================================================
# RUN THE APP
# ============================================================

if __name__ == "__main__":
    main()