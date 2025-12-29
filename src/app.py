import streamlit as st
from PIL import Image
import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
import time

# ---------- 1. PAGE SETUP ----------
st.set_page_config(
    page_title="Semantic Lens AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- 2. ULTRA-MODERN CSS (THE FIX) ----------
st.markdown("""
    <style>
    /* 1. FORCE DARK BACKGROUND & TEXT (Fixes Visibility) */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* 2. CENTERED CONTAINER FOR INPUTS */
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stForm"]) {
        max-width: 800px;
        margin: 0 auto;
    }

    /* 3. CUSTOM "GLASS" CARD FOR FORM */
    div[data-testid="stForm"] {
        background-color: #1e2329;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* 4. TYPOGRAPHY & TITLES */
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(0deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 10px !important;
    }
    h3 {
        color: #e6edf3 !important;
        font-weight: 600;
        font-size: 1.2rem;
    }
    p, label {
        color: #8b949e !important;
        font-size: 1rem;
    }

    /* 5. INPUT FIELDS (High Contrast) */
    .stTextInput > div > div > input {
        background-color: #0d1117;
        color: #ffffff;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2);
    }

    /* 6. ANALYZE BUTTON (Neon Gradient) */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        padding: 15px;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        margin-top: 15px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.01);
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.4);
    }

    /* 7. RESULT METRICS */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- 3. MODEL LOADING ----------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load BLIP
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    return clip_model, clip_processor, blip_model, blip_processor, device

# Load models quietly
clip_model, clip_processor, blip_model, blip_processor, device = load_models()

# ---------- 4. MAIN LAYOUT ----------
# Header Section
col_spacer1, col_content, col_spacer2 = st.columns([1, 2, 1])
with col_content:
    st.title("Semantic Lens")
    st.markdown(
        "<p style='text-align: center; margin-top: -15px; margin-bottom: 30px;'>AI-Powered Visual Consistency Verification</p>", 
        unsafe_allow_html=True
    )
    
    # Status Badge (Centered)
    status_color = "#238636" if device == "cuda" else "#d29922"
    status_text = "⚡ GPU ONLINE" if device == "cuda" else "🐢 CPU MODE"
    st.markdown(
        f"<div style='text-align: center; margin-bottom: 20px;'><span style='background-color: {status_color}; color: white; padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold;'>{status_text}</span></div>",
        unsafe_allow_html=True
    )

# Input Section (This will be styled as a card by the CSS)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.form("main_form"):
        st.markdown("### 1. Upload Source Image")
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        
        st.markdown("### 2. Enter Description")
        user_text = st.text_input("", placeholder="e.g. A futuristic city skyline at night...")
        
        submit = st.form_submit_button("✨ ANALYZE ALIGNMENT")

# ---------- 5. RESULTS SECTION ----------
if submit:
    if not uploaded_file or not user_text:
        st.warning("⚠️ Please provide both an image and a text description.")
    else:
        st.markdown("---")
        
        # Process Logic
        image = Image.open(uploaded_file).convert("RGB")
        
        # Layout for Results: Side-by-Side
        r_col1, r_col2 = st.columns([1, 1], gap="large")
        
        with r_col1:
            st.image(image, caption="Analyzed Image", use_container_width=True)
            
        with r_col2:
            with st.spinner("🔄 Neural networks are processing..."):
                time.sleep(0.5) # UX Pause
                
                # BLIP Captioning
                inputs = blip_processor(image, text="a photo of", return_tensors="pt").to(device)
                out = blip_model.generate(**inputs, max_new_tokens=40, num_beams=5, do_sample=False)
                ai_caption = blip_processor.decode(out[0], skip_special_tokens=True).replace("a photo of", "").strip()
                
                # CLIP Scoring
                clip_inputs = clip_processor(text=[user_text], images=image, return_tensors="pt", padding=True).to(device)
                outputs = clip_model(**clip_inputs)
                
                img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                score = (img_emb @ txt_emb[0].T).item()
                percentage = round(score * 100, 1)

            # Display Metrics
            if score >= 0.28:
                verdict = "MATCH CONFIRMED"
                color = "#238636" # Green
            elif score >= 0.20:
                verdict = "UNCERTAIN"
                color = "#d29922" # Orange
            else:
                verdict = "MISMATCH DETECTED"
                color = "#da3633" # Red

            st.markdown(f"### Consistency Score")
            st.metric(label="Semantic Alignment", value=f"{percentage}%")
            
            st.markdown(
                f"""
                <div style="background-color: {color}33; border-left: 5px solid {color}; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h4 style="margin:0; color: {color} !important;">{verdict}</h4>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown("### 🤖 AI Observation")
            st.info(f"The AI independently identified: **\"{ai_caption}\"**")
            
            # Progress Bar
            st.progress(min(max(score, 0), 1))