import subprocess
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from faster_whisper import WhisperModel
import asyncio
import edge_tts

# === Constants ===
AUDIO_INPUT = "input_audio.wav"
AUDIO_OUTPUT_MP3 = "response.mp3"
AUDIO_OUTPUT_WAV = "response_audio.wav"
VOICE = "en-US-AvaMultilingualNeural"

# === Speech-to-Text (STT) ===
def record_audio(duration=15):
    print("🎤 Listening...")
    samplerate = 16000
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(AUDIO_INPUT, recording, samplerate)

def transcribe_audio():
    model = WhisperModel("base", device="cpu")
    segments, _ = model.transcribe(AUDIO_INPUT)
    return " ".join([segment.text for segment in segments])

# === Text-to-Speech (TTS) ===
def synthesize_speech(text):
    async def speak():
        communicate = edge_tts.Communicate(text, voice=VOICE, device = "mps",rate="+4%")
        await communicate.save(AUDIO_OUTPUT_MP3)

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(speak())
        loop.run_until_complete(asyncio.gather(task))
    except RuntimeError:
        asyncio.run(speak())

    sound = AudioSegment.from_mp3(AUDIO_OUTPUT_MP3)
    sound.export(AUDIO_OUTPUT_WAV, format="wav")
    audio_data, samplerate = sf.read(AUDIO_OUTPUT_WAV)
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()

# === Main Loop ===
if __name__ == "__main__":
    intro = "Hello! I'm Shanthi, your AI therapist. I'm here to listen. How are you feeling today?"
    print("🧠 Shanthi:", intro)
    synthesize_speech(intro)

    while True:
        record_audio()
        user_input = transcribe_audio()
        print(f"You: {user_input}")

        if user_input.strip().lower() in ["exit", "quit", "bye"]:
            goodbye = "Take care of yourself. I'm always here if you need to talk."
            print("🧠 Shanthi:", goodbye)
            synthesize_speech(goodbye)
            break

        # Use LLaMA 3 via Ollama
        prompt = f"<s>[INST] You are Shanthi, a compassionate AI therapist. Be warm, brief, and reflective. Respond to: {user_input} [/INST]"
        result = subprocess.run(["ollama", "run", "llama3", prompt], capture_output=True, text=True)
        response = result.stdout.strip()

        print(f"🧠 Shanthi: {response}")
        synthesize_speech(response)
        print("🎤 Listening for your next message...")