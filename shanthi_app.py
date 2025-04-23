import streamlit as st
import asyncio
import sounddevice as sd
import soundfile as sf
import edge_tts
from faster_whisper import WhisperModel
import threading

# Initialize Whisper model for STT
whisper_model = WhisperModel("base", device="cpu")

# Initialize Edge-TTS settings
voice = "en-US-AvaMultilingualNeural"
rate = "+4%"
device = "mps"  # Use "mps" for Mac, "cpu" for CPU, or "cuda" for GPU

# Global conversation list
conversation = []

# Function to synthesize speech using edge-tts
async def synthesize_speech(text):
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate,device=device)
    await communicate.save("response.mp3")
    audio_data, samplerate = sf.read("response.mp3")
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()

# Function for transcribing speech to text (STT) using Whisper
def transcribe_audio(filename):
    segments, _ = whisper_model.transcribe(filename)
    return " ".join([segment.text for segment in segments])

# Add conversation to the chat
def add_to_conversation(user_input, bot_response):
    conversation.append(f"You: {user_input}")
    conversation.append(f"Shanthi: {bot_response}")

# Handle speech-to-text interaction
def handle_speech_interaction():
    with st.spinner("Listening..."):
        # Capture audio from the user
        user_input = transcribe_audio("input_audio.wav")
        st.session_state.user_input = user_input

        # Get response from Shanthi (this can be enhanced with RAG or direct query)
        bot_response = "I'm listening, tell me more."  # Dummy response, integrate with Shanthi's logic
        add_to_conversation(user_input, bot_response)

        # Display response and play audio
        st.text_area("Conversation", value="\n".join(conversation), height=300)
        asyncio.run(synthesize_speech(bot_response))

# Handle text-based interaction
def handle_text_interaction():
    user_input = st.text_input("You: ", "")
    if user_input:
        # Get response from Shanthi (this can be enhanced with RAG or direct query)
        bot_response = "I'm listening, tell me more."  # Dummy response, integrate with Shanthi's logic
        add_to_conversation(user_input, bot_response)
        
        # Display response and play audio
        st.text_area("Conversation", value="\n".join(conversation), height=300)
        asyncio.run(synthesize_speech(bot_response))

# Streamlit frontend setup
def run_frontend():
    st.title("Shanthi: Your AI Therapist")
    st.write("Welcome! Choose between speech-based or text-based interactions.")
    
    # Add buttons to switch between interactions
    interaction_mode = st.radio("Choose interaction mode", ("Speech", "Text"))
    
    if interaction_mode == "Speech":
        handle_speech_interaction()
    elif interaction_mode == "Text":
        handle_text_interaction()

if __name__ == "__main__":
    # Start the Streamlit app
    run_frontend()
