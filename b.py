import streamlit as st
from transformers import pipeline
from PIL import Image
import os

# Set page title and icon
st.set_page_config(
    page_title="Facial Emotion Analyzer",
    page_icon="😊",
    layout="centered"
)

# Cache model loading (important! Avoid repeated loading)
@st.cache_resource
def load_model():
    return pipeline("image-classification", 
                   model="dima806/facial_emotions_image_detection")

# Initialize the model
model = load_model()

# Define corresponding coping methods for different emotions
emotion_advice = {
    "neutral": "The user is in a good state. Keep up the current rhythm of life and mindset, and engage in some relaxing activities to unwind.",
    "happy": "You're in a great mood! Share this joy with friends and family, and record the wonderful moments to double the happiness.",
    "sad": "It seems you're feeling a bit sad. Talk to someone close to you, or listen to some soothing music or watch a comedy movie to ease your mood.",
    "angry": "It looks like you're angry. Take a few deep breaths to calm down. You can also find an open space to shout and release the stress.",
    "surprise": "You've encountered a pleasant surprise! Enjoy this unexpected happiness and share the joy with those around you.",
    "fear": "If you're feeling scared, stay in a safe environment and be with someone you trust to feel more secure.",
    "disgust": "If you're feeling disgusted, try to stay away from the things that make you feel this way and do something that makes you happy to change your mood."
}

# Page title
st.title("😄 Facial Emotion Recognition System")
st.markdown("---")

# Sidebar instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload an image containing a human face (JPG/PNG/GIF).
    2. The system will automatically detect facial emotions.
    3. View the analysis results and confidence levels.
    """)
    st.markdown("---")
    st.caption("Technical support: Hugging Face Transformers")

# File upload area
uploaded_file = st.file_uploader(
    "Please select an image file",
    type=["jpg", "jpeg", "png", "gif"],
    help="Supported formats: JPG/PNG/GIF"
)

if uploaded_file is not None:
    try:
        # Display the uploaded image
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)

            # If it's a GIF, display the first frame by default
            if getattr(image, "is_animated", False):
                image.seek(0)
                st.image(image, use_container_width=True)
                st.caption("A GIF animation was detected. The first frame is used for analysis by default.")
            else:
                st.image(image, use_container_width=True)

        # Analysis button
        analyze_button = st.button("Start Analysis")
        if analyze_button:
            with col2:
                st.subheader("Analysis Results")
                with st.spinner("Analyzing facial emotions..."):
                    # Make predictions (automatically use the first frame for GIFs)
                    results = model(image)

                    # Display the main emotion
                    main_emotion = results[0]['label']
                    confidence = results[0]['score'] * 100
                    st.metric(
                        label="Main Emotion",
                        value=f"{main_emotion}",
                        delta=f"{confidence:.1f}% confidence"
                    )

                    # Display the detailed probability distribution
                    st.markdown("**Emotion Distribution:**")
                    for result in results:
                        # Display with a progress bar
                        st.progress(
                            value=result['score'],
                            text=f"{result['label'].capitalize()}: {result['score'] * 100:.1f}%"
                        )

                    # Display the corresponding coping method for the emotion
                    if main_emotion in emotion_advice:
                        st.markdown("### Coping Advice")
                        st.write(emotion_advice[main_emotion])
                    else:
                        st.markdown("### Coping Advice")
                        st.write(f"Sorry, no advice available for the emotion: {main_emotion}")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.stop()

# Add a footer
st.markdown("---")
st.caption("Note: This system uses a deep learning-based emotion recognition model, and the analysis results are for reference only.")