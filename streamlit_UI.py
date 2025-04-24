import streamlit as st
import time
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import asyncio
import edge_tts
import subprocess
from pydub import AudioSegment
import os

# === Constants ===
AUDIO_INPUT = "input_audio.wav"
AUDIO_OUTPUT_MP3 = "response.mp3"
AUDIO_OUTPUT_WAV = "response_audio.wav"
VOICE = "en-US-AvaMultilingualNeural"

st.set_page_config(page_title="Shanthi - Voice Therapist", layout="centered", initial_sidebar_state="collapsed")

# Apply dark mode theme
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .stChatMessage { margin-bottom: 1rem; }
    .shanthibubble {
        background-color: #2c2c2c;
        padding: 1rem;
        border-radius: 20px;
        max-width: 80%;
        margin-bottom: 10px;
    }
    .userbubble {
        background-color: #3a3a3a;
        padding: 1rem;
        border-radius: 20px;
        max-width: 80%;
        margin-left: auto;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Whisper transcription
model = WhisperModel("base", device="cpu")
def record_audio(duration=15):
    samplerate = 16000
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(AUDIO_INPUT, recording, samplerate)

def transcribe_audio():
    segments, _ = model.transcribe(AUDIO_INPUT)
    return " ".join([segment.text for segment in segments])

# TTS
async def synthesize_speech(text):
    communicate = edge_tts.Communicate(text, voice=VOICE, rate="+5%")
    await communicate.save(AUDIO_OUTPUT_MP3)
    sound = AudioSegment.from_mp3(AUDIO_OUTPUT_MP3)
    sound.export(AUDIO_OUTPUT_WAV, format="wav")
    audio_data, samplerate = sf.read(AUDIO_OUTPUT_WAV)
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()

# Shanthi prompts
SYSTEM_PROMPT = """
You are Shanthi, a compassionate AI therapist trained in CBT and motivational interviewing. Your job is not to solve problems, but to help users understand their emotions and discover insights on their own.
When a user says something emotional, follow this flow:
1. Empathize and validate their feelings.
2. Ask gentle, open-ended follow-up questions.
3. Do not give suggestions unless they ask for them.
4. Speak in a conversational, friendly English tone, depending on user language.
5. Keep responses short and warm, not robotic or verbose.
Always end your message with a reflective or clarifying question to keep the conversation going.
"""

def detect_emotional_state(user_input: str):
    user_input = user_input.lower()
    if any(phrase in user_input for phrase in ["don’t want to talk", "don’t feel like talking", "leave me alone"]):
        return "silent"
    elif any(word in user_input for word in ["sad", "depressed", "tired", "anxious"]):
        return "low"
    elif any(word in user_input for word in ["breakup", "relationship", "alone", "losing my job"]):
        return "multiple_problems"
    return "open"

def get_system_prompt(state: str):
    if state == "silent":
        return "You are Shanthi, a caring AI therapist. The user doesn't want to talk right now. Gently reassure them without asking questions."
    elif state == "low":
        return "You are Shanthi, a caring AI therapist. The user is sad or stressed. Speak softly and gently. Be supportive, not too wordy. Avoid solutions unless asked."
    elif state == "multiple_problems":
        return "You are Shanthi, a caring AI therapist. The user has multiple emotional issues. Acknowledge their feelings, empathize with each problem, and gently explore them one by one."
    else:
        return "You are Shanthi, an empathetic AI therapist who speaks in casual English. Keep it warm, modern, short, and inviting. Don’t push, reflect emotions."

def query_llama(user_input, emotional_state):
    system_prompt = get_system_prompt(emotional_state)
    full_prompt = f"[INST] <<SYS>> {system_prompt} <</SYS>>\n{user_input} [/INST]"
    result = subprocess.run(["ollama", "run", "llama3", full_prompt], capture_output=True, text=True)
    response = result.stdout.strip()
    return response.replace("<s>", "").replace("</s>", "").strip()

# === Streamlit Chat App ===
if "chat" not in st.session_state:
    st.session_state.chat = []
    st.session_state.first_run = True

st.title("🧠 Shanthi - Your Voice Companion")

chat_placeholder = st.empty()

# UI rendering
with chat_placeholder.container():
    for sender, message in st.session_state.chat:
        if sender == "user":
            st.markdown(f'<div class="userbubble">🧍‍♂️ {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="shanthibubble">🧠 {message}</div>', unsafe_allow_html=True)

# === Main Conversation Loop ===
def chat_loop():
    while True:
        record_audio()
        user_input = transcribe_audio()
        if user_input.strip() == "":
            continue
        st.session_state.chat.append(("user", user_input))
        chat_placeholder.empty()
        with chat_placeholder.container():
            for sender, message in st.session_state.chat:
                if sender == "user":
                    st.markdown(f'<div class="userbubble">🧍‍♂️ {message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="shanthibubble">🧠 {message}</div>', unsafe_allow_html=True)

        state = detect_emotional_state(user_input)
        response = query_llama(user_input, state)
        st.session_state.chat.append(("shanthi", response))
        with chat_placeholder.container():
            for sender, message in st.session_state.chat:
                if sender == "user":
                    st.markdown(f'<div class="userbubble">🧍‍♂️ {message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="shanthibubble">🧠 {message}</div>', unsafe_allow_html=True)
        asyncio.run(synthesize_speech(response))

if st.session_state.first_run:
    intro = "Hello! I'm Shanthi, your AI therapist. I'm here to listen. How are you feeling today?"
    st.session_state.chat.append(("shanthi", intro))
    with chat_placeholder.container():
        st.markdown(f'<div class="shanthibubble">🧠 {intro}</div>', unsafe_allow_html=True)
    asyncio.run(synthesize_speech(intro))
    st.session_state.first_run = False

# Start the loop automatically after first response
chat_loop()
