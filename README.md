# Shnathi 🧠 – Empathetic AI Therapist

**Shnathi** is a locally running voice-based AI therapist built with LLaMA3, Whisper (for speech-to-text), and Edge-TTS (for text-to-speech). Inspired by CBT and motivational interviewing, Shnathi offers warm, non-judgmental conversation in English and Swedish.

---

## 🚀 Features

- 🗣️ Voice-based and 💬 text-based interactions
- 🧘 Empathetic, CBT-style dialogue logic
- 🔊 Local speech-to-text (Whisper) and text-to-speech (Edge-TTS)
- 🧠 Locally running LLaMA3-based model via Ollama
- 📂 Organized for easy model fine-tuning and frontend extension

---

## 🗂️ Directory Structure

Shnathi/ ├── app/ │ ├── shanthi_app.py # Main application logic │ └── frontend.js # (WIP) Web interface script ├── audio/ │ ├── input_audio.wav # Temporary audio file for STT │ └── response_audio.wav # Temporary audio file for TTS ├── model/ │ ├── finetune_shanthi.py # Fine-tuning LLaMA or other models │ └── rag_shanthi.py # RAG pipeline logic ├── data/ │ ├── data_collection.py │ ├── data_preprocessing.py │ ├── empathetic_dialogues.py │ ├── extract_fb_emp_conv.py │ └── empathetic_dialogues/ # Dialogue dataset files ├── utils/ │ ├── ollama_utils.py # Whisper and Edge-TTS helpers │ ├── tone_finder.py # Tone detection logic │ └── tone_finder_bert.py # (Optional) tone using BERT ├── requirements.txt # Python dependencies └── README.md # You're here!

yaml
Copy
Edit

---

## 🛠️ Setup Instructions

### 1. Clone the repo


---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/siddharthsaminathan/Shnathi.git
cd Shnathi


🎯 Next Steps
 Add frontend toggle between text and voice

 Fine-tune LLaMA3 on emotional/empathetic data

 Add persistent memory support

 Create exportable transcripts per session

🧡 Acknowledgements
OpenAI Whisper

Edge-TTS (Microsoft)

Meta LLaMA3

Hugging Face Datasets (Empathetic Dialogues)

📬 Contact
Siddharth Saminathan
📫 siddharthsaminathan99@gmail.com
🌍 LinkedIn
