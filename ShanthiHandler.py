# 🧠 Shanthi: Emotionally aware AI Therapist (English only)
# Version: Dynamic, Memory-Aware, State-Adaptive

import random
from ollama_talk import query_llama3, synthesize_speech, transcribe_audio  # Updated imports to use functions from ollama_talk

# Sample tone variations by emotional state
RESPONSES = {
    "initial": [
        "Hi there. I'm here for you—how are you feeling today?",
        "Thanks for reaching out. What would you like to talk about?"
    ],
    "fragile": [
        "Thank you for opening up about this. It's heavy, I know. But saying it out loud really matters.",
        "I hear you. Let's take it slow, one step at a time."
    ],
    "open": [
        "Sounds like you've been reflecting on this. Want to tell me more about how it's affecting you right now?",
        "It's strong of you to share. What do you usually feel when this happens?"
    ],
    "low_energy": [
        "It's okay to feel drained sometimes. We don’t have to fix anything right now. Want to just sit together a bit?",
        "That sounds heavy. Would you like silence for a bit, or should I ask something gentle?"
    ],
    "relapsing": [
        "This sounds like a tough moment. Want to look at what might be triggering this together?",
        "Nobody's perfect. You're here, you're trying. That matters."
    ]
}

class Shanthi:
    def __init__(self):
        self.history = []
        self.state = "initial"

    def determine_state(self, user_input):
        text = user_input.lower()
        if any(w in text for w in ["hopplös", "orkar inte", "tom", "trött", "vet inte"]):
            return "low_energy"
        if any(w in text for w in ["beroende", "porr", "missbruk"]):
            return "fragile"
        if len(self.history) >= 3 and "inte prata" not in text:
            return "open"
        if any(w in text for w in ["återfall", "igen", "misslyckats"]):
            return "relapsing"
        return "initial"

    def respond(self, user_input):
        self.history.append(user_input)
        self.state = self.determine_state(user_input)

        if self.state in ["fragile", "low_energy"]:
            # Use predefined empathetic responses for fragile states
            return random.choice(RESPONSES[self.state])
        else:
            # Use the model from ollama_utils.py for open or complex queries
            return query_llama3(user_input)

    def interactive_session(self):
        print("Shanthi: Hello! I am Shanthi, your AI therapist. How can I help you today?")
        synthesize_speech("Hello! I am Shanthi, your AI therapist. How can I help you today?")

        while True:
            print("Listening...")
            user_input = transcribe_audio("input_audio.wav")
            print(f"You: {user_input}")

            if user_input.lower() in ["exit", "quit"]:
                print("Shanthi: Goodbye!")
                synthesize_speech("Goodbye!")
                break

            response = self.respond(user_input)
            print(f"Shanthi: {response}")
            synthesize_speech(response)

# --- Example usage ---
if __name__ == "__main__":
    shanthi = Shanthi()
    shanthi.interactive_session()
