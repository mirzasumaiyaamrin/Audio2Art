import streamlit as st
import wave
import numpy as np
import torch
from transformers import pipeline
import openai
import time

# Set OpenAI API Key (Replace with your actual API key)
openai.api_key = "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"

# Load Whisper model locally using transformers
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

# Function to upload audio instead of recording
def upload_audio():
    st.info("🔊 Please upload a recorded voice file (WAV format).")
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
    if audio_file:
        st.audio(audio_file, format="audio/wav")
        return audio_file
    return None

# Function to transcribe audio using Hugging Face Transformers (Whisper)
def transcribe_audio(audio_file):
    """Transcribes speech locally using transformers' Whisper model."""
    with st.spinner("📝 Transcribing audio..."):
        result = whisper_pipeline(audio_file)
    return result.get("text", "⚠️ Could not transcribe the audio.")

# Function to generate an image using OpenAI DALL·E API
def generate_image(prompt):
    """Generates an image using OpenAI's DALL·E API."""
    with st.spinner("🎨 Generating AI Art... Please wait."):
        response = openai.images.generate(
            model="dall-e-2",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
    
    if hasattr(response, "data"):
        image_url = response.data[0].url
        st.image(image_url, caption="🎨 Generated Image", use_container_width=True)
    else:
        st.error("⚠️ Error generating image.")

# Streamlit UI
def main():
    """Streamlit UI for Audio2Art using OpenAI APIs & Transformers."""
    st.set_page_config(page_title="Audio2Art", page_icon="🎨", layout="wide")
    
    st.sidebar.title("🔊 Audio2Art Settings")
    st.sidebar.write("🎙️ Upload a description, and AI will create an image!")

    st.title("🎨 Audio2Art: Voice to AI Art")
    st.write("Transform your voice into stunning AI-generated artwork! Upload a recorded voice file, let AI transcribe it, and watch your words turn into visuals.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        audio_file = upload_audio()
        if audio_file:
            st.success("✅ Audio uploaded! Transcribing...")
            prompt = transcribe_audio(audio_file)
            st.success(f"📝 You said: {prompt}")

            with st.spinner("🎨 Generating AI Art..."):
                time.sleep(1)  # Adding a slight delay for better UI experience
                generate_image(prompt)
    
    with col2:
        st.image("https://picsum.photos/1024/1024", caption="🎭 AI Creativity", use_container_width=True)

if __name__ == "__main__":
    main()
