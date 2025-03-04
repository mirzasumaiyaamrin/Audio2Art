import streamlit as st
import torch
from transformers import pipeline
import openai
import tempfile
import time
import os

# ✅ Ensure Streamlit Page Config is the FIRST command
st.set_page_config(page_title="Audio2Art", page_icon="🎨", layout="wide")

# ✅ Load OpenAI API Key securely from environment variable
openai_api_key = st.secrets.get("OPENAI_API_KEY")  # Fetch API key safely
if not openai_api_key:
    st.error("⚠️ OpenAI API key is missing! Add it in Streamlit Secrets.")
    st.stop()

# ✅ Set OpenAI API Key (Fixed)
openai.api_key = openai_api_key

# ✅ Load Whisper model for speech recognition
device = "cpu"  # Force CPU mode for better compatibility on Streamlit Cloud
whisper_pipeline = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en", device=device)

# ✅ Function: Transcribe audio using Hugging Face Transformers (Whisper)
def transcribe_audio(uploaded_audio):
    """Transcribes speech from an uploaded audio file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_audio.read())  # Save uploaded file
            temp_audio_path = temp_audio.name

        with st.spinner("📝 Transcribing audio..."):
            result = whisper_pipeline(temp_audio_path)

        return result.get("text", "⚠️ Could not transcribe the audio.")
    
    except Exception as e:
        st.error(f"❌ Error transcribing audio: {e}")
        return None

# ✅ Function: Generate image using OpenAI DALL·E API
def generate_image(prompt):
    """Generates an image using OpenAI's DALL·E API."""
    try:
        with st.spinner("🎨 Generating AI Art... Please wait."):
            response = openai.Image.create(  # ✅ Fixed OpenAI API Call
                prompt=prompt,
                n=1,
                size="1024x1024"
            )

        if "data" in response:
            image_url = response["data"][0]["url"]  # ✅ Fixed response handling
            st.image(image_url, caption="🎨 Generated Image", use_container_width=True)
        else:
            st.error("⚠️ No image received from OpenAI.")

    except Exception as e:
        st.error(f"❌ Image generation failed: {e}")

# ✅ Function: Load Placeholder Image (Fix for External Image Loading)
def load_placeholder_image():
    """Loads a placeholder image in case of failure."""
    try:
        st.image("https://picsum.photos/1024/1024", caption="🎭 AI Creativity", use_container_width=True)
    except Exception as e:
        st.warning("⚠️ Could not load the placeholder image.")
        st.write(f"Error: {e}")

# ✅ Streamlit UI
def main():
    """Streamlit UI for Audio2Art using OpenAI & Transformers."""
    st.sidebar.title("🔊 Audio2Art Settings")
    st.sidebar.write("🎙️ Upload an audio file to generate AI art!")

    st.title("🎨 Audio2Art: Voice to AI Art")
    st.write("Transform your voice into AI-generated artwork! Upload an audio file, let AI transcribe it, and turn your words into images.")

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_audio = st.file_uploader("🎤 Upload an audio file (WAV, MP3)", type=["wav", "mp3"])

        if uploaded_audio:
            st.audio(uploaded_audio, format="audio/wav")
            st.success("✅ Audio uploaded! Transcribing...")

            prompt = transcribe_audio(uploaded_audio)

            if prompt:
                st.success(f"📝 You said: {prompt}")
                
                with st.spinner("🎨 Generating AI Art..."):
                    time.sleep(1)  # Slight delay for better UI experience
                    generate_image(prompt)
            else:
                st.error("⚠️ Could not transcribe the audio. Please try again.")

    with col2:
        load_placeholder_image()

if __name__ == "__main__":
    main()
