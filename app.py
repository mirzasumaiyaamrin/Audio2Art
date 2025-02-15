#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
import streamlit as st
import sounddevice as sd
import wave
import numpy as np
import torch
from transformers import pipeline
import openai
from tempfile import NamedTemporaryFile
import time

# Set OpenAI API Key (Replace with your actual API key)
openai.api_key = "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"

# Load Whisper model locally using transformers
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

# Function to record audio
def record_audio(duration=5, samplerate=44100):
    """Records audio and saves it as a temporary WAV file."""
    with st.spinner("🎤 Recording... Speak now!"):
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
    
    temp_audio = NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_audio.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    return temp_audio.name

# Function to transcribe audio using Hugging Face Transformers (Whisper)
def transcribe_audio(audio_path):
    """Transcribes speech locally using transformers' Whisper model."""
    with st.spinner("📝 Transcribing audio..."):
        result = whisper_pipeline(audio_path)
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
    duration = st.sidebar.slider("🎤 Recording Duration (seconds)", 3, 10, 5)
    st.sidebar.write("🎙️ Speak a description, and AI will create an image!")
    
    st.title("🎨 Audio2Art: Voice to AI Art")
    st.write("Transform your voice into stunning AI-generated artwork! Record your voice, let AI transcribe it, and watch your words turn into visuals.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🎤 Start Recording", use_container_width=True):
            audio_path = record_audio(duration)
            st.success("✅ Recording complete! Transcribing...")
            
            prompt = transcribe_audio(audio_path)
            st.success(f"📝 You said: {prompt}")
            
            with st.spinner("🎨 Generating AI Art..."):
                time.sleep(1)  # Adding a slight delay for better UI experience
                generate_image(prompt)
    
    with col2:
        st.image("https://picsum.photos/1024/1024", caption="🎭 AI Creativity", use_container_width=True)

if __name__ == "__main__":
    main()
