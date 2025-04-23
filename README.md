# Shanthi AI Therapist

Shanthi is an AI-powered therapist designed to provide empathetic, supportive conversations based on Cognitive Behavioral Therapy (CBT) and Motivational Interviewing techniques. Built with a combination of machine learning, natural language processing, and voice synthesis technologies, Shanthi helps users explore their thoughts and emotions in a compassionate and non-judgmental way.

## Features

- **Conversational AI**: Shanthi engages in empathetic and reflective conversations with users.
- **Voice Interaction**: Supports both Speech-to-Text (STT) and Text-to-Speech (TTS) using local models.
- **Emotion Detection**: Able to detect and respond to different emotional states, making the conversation feel more personal and supportive.
- **Multilingual Support**: Currently supports English and Swedish, with more languages to be added in future versions.

## Technologies Used

- **Machine Learning**: 
  - Fine-tuned models like Mistral and LLaMA for empathetic dialogue generation.
  - Whisper for Speech-to-Text.
  - Edge TTS for Text-to-Speech (local TTS engine).
- **Backend**: Python-based backend with integrated AI models.
- **Frontend**: The system can operate with either text-based or voice-based interactions, with flexibility for integration into different interfaces.
- **Libraries**: 
  - `transformers`, `torch` for machine learning models.
  - `sounddevice`, `soundfile` for audio playback and recording.
  - `edge-tts` for speech synthesis.
  
## Getting Started

### Prerequisites

1. **Python** (>= 3.8)
2. **Virtual Environment**: It is recommended to create a virtual environment for dependencies.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/siddharthsaminathan/Shanthi_AI_Therapist.git
   cd Shanthi_AI_Therapist
   
2. Set up a virtual environment and activate it:
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   .\venv\Scripts\activate   # On Windows
   
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

4. Download the necessary AI models:

Whisper for Speech-to-Text

Edge TTS for Text-to-Speech

LLaMA/Mistral model for conversational AI (specific instructions are in the repo for model setup). -  HuggingFace or Ollama for mac with mps support

## Running Shanthi
Run the script to start the interaction:

   ```bash
   python ollama_talk.py

```
Engage with Shanthi! It will respond based on the emotional context and your tone. 

## Frontend (Optional)
If you're interested in running a frontend for Shanthi, there are options for both text-based or voice-based interactions. The frontend allows you to switch between speech and text, and it displays the conversation history.

For a simple local web interface, you can refer to the streamlit_UI.py file to run it with Streamlit, or build your own custom interface.

## Future Plans
Expand Emotional Detection: Enhance emotion detection and response generation for better conversation quality.

Multilingual Support: Add more languages to cater to a wider audience.

Mobile App Integration: Make Shanthi accessible on mobile platforms.

## Contributing
Feel free to fork the repository, submit issues, and make pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
OpenAI for GPT-based models.

Whisper and Edge TTS for local speech processing.

Hugging Face for model-sharing and community support.





