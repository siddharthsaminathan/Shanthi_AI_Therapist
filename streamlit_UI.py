import gradio as gr
from faster_whisper import WhisperModel
import requests

# Initialize Faster-Whisper model for STT
@gr.cache()
def load_whisper_model():
    return WhisperModel("base", device="cpu")  # Use "cuda" if GPU is available

whisper_model = load_whisper_model()

# LLaMA3 prompt template
SYSTEM_PROMPT = (
    "You are Shanthi, a compassionate AI therapist trained in CBT and motivational interviewing."
    " Your job is not to solve problems, but to help users understand their emotions and discover insights on their own."
    "\n\nWhen a user says something emotional, follow this flow:"
    "\n1. Empathize and validate their feelings."
    "\n2. Ask gentle, open-ended follow-up questions."
    "\n3. Do not give suggestions unless they ask for them."
    "\n4. Speak in a conversational, friendly Swedish or English tone, depending on user language."
    "\n5. Keep responses short and warm, not robotic or verbose."
    "\n\nAlways end your message with a reflective or clarifying question to keep the conversation going."
)

# LLaMA3 local call via Ollama API
def query_llama3(prompt):
    """Send a prompt to the LLaMA3 model via Ollama API."""
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": f"<|system|>{SYSTEM_PROMPT}<|user|>{prompt}<|assistant|>",
        "stream": False
    })
    if response.ok:
        return response.json().get("response", "")
    else:
        return "Oops! Something went wrong with the model response."

# Gradio interface
def chat_with_shanthi(user_input):
    if user_input:
        response = query_llama3(user_input)
        return response
    return "Please enter a message to start the conversation."

iface = gr.Interface(
    fn=chat_with_shanthi,
    inputs=gr.Textbox(lines=2, placeholder="How are you feeling today?"),
    outputs=gr.Textbox(lines=4, placeholder="Shanthi's response will appear here."),
    title="Shanthi - AI Therapist",
    description="Chat with Shanthi, your AI therapist trained in CBT and motivational interviewing."
)

if __name__ == "__main__":
    iface.launch()