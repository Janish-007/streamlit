import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import io
import requests
from typing import List, Tuple

# Configure page
st.set_page_config(
    page_title="CLIP Classifier",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_clip_model():
    """Load CLIP model and preprocessing function"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device
    except Exception as e:
        st.error(f"Error loading CLIP model: {e}")
        return None, None, None

def classify_input(model, preprocess, device, input_data, positive_prompts, negative_prompts, input_type="image"):
    """
    Classify input based on positive and negative prompts using CLIP
    """
    try:
        # Prepare text prompts
        all_prompts = positive_prompts + negative_prompts
        text_inputs = clip.tokenize(all_prompts).to(device)
        
        if input_type == "image":
            # Process image
            if isinstance(input_data, str):  # URL
                response = requests.get(input_data)
                image = Image.open(io.BytesIO(response.content))
            else:  # Uploaded file
                image = Image.open(input_data)
            
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Get features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                similarities = similarities[0].cpu().numpy()
        
        elif input_type == "text":
            # Process text input
            input_text = clip.tokenize([input_data]).to(device)
            
            with torch.no_grad():
                input_features = model.encode_text(input_text)
                text_features = model.encode_text(text_inputs)
                
                # Calculate similarities
                similarities = (100.0 * input_features @ text_features.T).softmax(dim=-1)
                similarities = similarities[0].cpu().numpy()
        
        # Calculate scores for positive and negative categories
        positive_scores = similarities[:len(positive_prompts)]
        negative_scores = similarities[len(positive_prompts):]
        
        positive_total = np.sum(positive_scores)
        negative_total = np.sum(negative_scores)
        
        # Determine classification
        is_positive = positive_total > negative_total
        confidence = max(positive_total, negative_total)
        
        return {
            'classification': 'Positive' if is_positive else 'Negative',
            'confidence': float(confidence),
            'positive_score': float(positive_total),
            'negative_score': float(negative_total),
            'detailed_scores': {
                'positive_prompts': [(prompt, float(score)) for prompt, score in zip(positive_prompts, positive_scores)],
                'negative_prompts': [(prompt, float(score)) for prompt, score in zip(negative_prompts, negative_scores)]
            }
        }
    
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

def main():
    st.title("🔍 CLIP-Based Custom Classifier")
    st.markdown("### Define your own positive and negative prompts to classify images or text!")
    
    # Load model
    model, preprocess, device = load_clip_model()
    
    if model is None:
        st.error("Failed to load CLIP model. Please check your installation.")
        st.stop()
    
    st.success(f"CLIP model loaded successfully on {device}")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Input type selection
        input_type = st.radio("Select input type:", ["Image", "Text"])
        
        st.header("📝 Define Prompts")
        
        # Positive prompts
        st.subheader("✅ Positive Prompts")
        positive_prompts_text = st.text_area(
            "Enter positive prompts (one per line):",
            value="happy face\nsmiling person\njoyful expression\npositive emotion",
            height=100,
            help="These prompts define what should be classified as 'Positive'"
        )
        
        # Negative prompts
        st.subheader("❌ Negative Prompts")
        negative_prompts_text = st.text_area(
            "Enter negative prompts (one per line):",
            value="sad face\nangry person\nfrowning expression\nnegative emotion",
            height=100,
            help="These prompts define what should be classified as 'Negative'"
        )
        
        # Process prompts
        positive_prompts = [p.strip() for p in positive_prompts_text.split('\n') if p.strip()]
        negative_prompts = [p.strip() for p in negative_prompts_text.split('\n') if p.strip()]
        
        st.info(f"Positive prompts: {len(positive_prompts)}")
        st.info(f"Negative prompts: {len(negative_prompts)}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📥 Input")
        
        input_data = None
        
        if input_type == "Image":
            # Image input options
            image_option = st.radio("Choose image source:", ["Upload", "URL"])
            
            if image_option == "Upload":
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
                )
                if uploaded_file:
                    input_data = uploaded_file
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            else:  # URL
                image_url = st.text_input("Enter image URL:")
                if image_url:
                    try:
                        response = requests.get(image_url)
                        image = Image.open(io.BytesIO(response.content))
                        input_data = image_url
                        st.image(image, caption="Image from URL", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image from URL: {e}")
        
        else:  # Text input
            text_input = st.text_area(
                "Enter text to classify:",
                height=150,
                placeholder="Type your text here..."
            )
            if text_input.strip():
                input_data = text_input.strip()
                st.text_area("Text to classify:", value=text_input, height=100, disabled=True)
    
    with col2:
        st.header("📊 Results")
        
        if input_data and positive_prompts and negative_prompts:
            if st.button("🚀 Classify", type="primary", use_container_width=True):
                with st.spinner("Classifying..."):
                    result = classify_input(
                        model, preprocess, device, input_data, 
                        positive_prompts, negative_prompts,
                        input_type.lower()
                    )
                
                if result:
                    # Main classification result
                    classification = result['classification']
                    confidence = result['confidence']
                    
                    # Display result with color coding
                    color = "green" if classification == "Positive" else "red"
                    st.markdown(f"### Classification: <span style='color: {color}'>{classification}</span>", 
                              unsafe_allow_html=True)
                    
                    # Confidence and scores
                    st.metric("Confidence", f"{confidence:.3f}")
                    
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.metric("Positive Score", f"{result['positive_score']:.3f}")
                    with col_neg:
                        st.metric("Negative Score", f"{result['negative_score']:.3f}")
                    
                    # Detailed breakdown
                    st.subheader("📈 Detailed Scores")
                    
                    # Positive prompts scores
                    st.write("**Positive Prompts:**")
                    for prompt, score in result['detailed_scores']['positive_prompts']:
                        st.progress(float(score), text=f"{prompt}: {score:.3f}")
                    
                    # Negative prompts scores
                    st.write("**Negative Prompts:**")
                    for prompt, score in result['detailed_scores']['negative_prompts']:
                        st.progress(float(score), text=f"{prompt}: {score:.3f}")
        
        elif not positive_prompts or not negative_prompts:
            st.warning("⚠️ Please define both positive and negative prompts in the sidebar.")
        
        elif not input_data:
            st.info("📝 Please provide input data to classify.")
    
    # Instructions
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Define Prompts**: In the sidebar, enter your positive and negative prompts (one per line)
        2. **Choose Input Type**: Select whether you want to classify images or text
        3. **Provide Input**: 
           - For images: Upload a file or provide a URL
           - For text: Type or paste your text
        4. **Classify**: Click the "Classify" button to see results
        
        **Examples of prompts:**
        - **Image classification**: "happy dog, playful pet" vs "aggressive dog, angry animal"
        - **Text sentiment**: "positive review, good experience" vs "negative review, bad experience"
        - **Content moderation**: "safe content, family friendly" vs "inappropriate content, offensive material"
        """)

if __name__ == "__main__":
    main()