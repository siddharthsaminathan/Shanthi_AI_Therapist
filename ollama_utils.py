import subprocess
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import edge_tts
import asyncio

AUDIO_FILE = "response_audio.wav"
INPUT_AUDIO_FILE = "input_audio.wav"

# === Text-to-Speech ===
async def synthesize_speech(text, voice="en-IN-NeerjaNeural"):
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(AUDIO_FILE)
    audio_data, samplerate = sf.read(AUDIO_FILE)
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()

# === Speech-to-Text ===
def transcribe_audio(filename):
    model = WhisperModel("base", device="cpu")
    segments, _ = model.transcribe(filename)
    return " ".join([segment.text for segment in segments])

# === Main Loop ===
async def main():
    intro = "Hello! I am Shanthi, your AI therapist. How can I help you today?"
    print(f"Shanthi: {intro}")
    await synthesize_speech(intro)

    while True:
        print("Listening...")
        user_input = transcribe_audio(INPUT_AUDIO_FILE)
        print(f"You: {user_input}")

        if user_input.strip().lower() in ["exit", "quit"]:
            goodbye = "Take care until we meet again."
            print(f"Shanthi: {goodbye}")
            await synthesize_speech(goodbye)
            break

        # Construct prompt for LLaMA 3
        system_prompt = "You are Shanthi, a warm and compassionate AI therapist. Help the user reflect and understand themselves without giving direct advice."
        full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_input} [/INST]"

        result = subprocess.run(
            ["ollama", "run", "llama3", full_prompt],
            capture_output=True,
            text=True
        )

        response = result.stdout.strip()
        print(f"Shanthi: {response}")
        await synthesize_speech(response)

if __name__ == "__main__":
    asyncio.run(main())
