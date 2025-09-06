from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
from gtts import gTTS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# --- Gemini API config ---
GEMINI_API_KEY = "AIzaSyBv9GHyMba0gy28ALwa9hpQNWpxAoUyB5g"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"  # You can also use "gemini-2.5-flash" if available


PROMPT = "You are a helpful assistant and an agricultural bot who gives summarized analysis no symbols strictly only in maximum of 2 or 3 senteces ! " \
"cuz i m going to use for voice and crop suggestion or like what crop to plant do analysis on the given data," \
" in either Tamil or English based on user input. " \
"you should also provide current price predictions for the crops suggested in rupees. DONT TALK TOO MUCH NO ONE LIKE ANYONE WHO TALK TOO MUCH"

# --- Language detection ---
def detect_language(text: str) -> str:
    """Detect Tamil vs English based on Unicode range"""
    for ch in text:
        if 0x0B80 <= ord(ch) <= 0x0BFF:
            return "ta"
    return "en"

# --- Text to speech using gTTS ---
def text_to_speech(content: str) -> str:
    lang = detect_language(content)
    tts = gTTS(text=content, lang=lang)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return base64.b64encode(audio_buffer.read()).decode("utf-8")

# --- Gemini API call function ---
def call_gemini_api(messages):
    """Call Gemini API with the provided messages"""
    try:
        # Format messages for Gemini API
        # Gemini doesn't use a system message in the same way, so we'll include it as the first user message
        formatted_content = messages[0]["content"] if messages[0]["role"] == "system" else ""
        
        for msg in messages:
            if msg["role"] == "system" and formatted_content == "":
                formatted_content = msg["content"]
            elif msg["role"] == "user":
                formatted_content += f"\n\nUser: {msg['content']}"
            elif msg["role"] == "assistant":
                formatted_content += f"\n\nAssistant: {msg['content']}"
        
        # Initialize the model
        model = genai.GenerativeModel(MODEL)
        
        # Generate content
        response = model.generate_content(formatted_content)
        
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Currently unavailable due to an internal error."

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    user_content = data.get("content", "")
    language = data.get("language", "en")

    if language == "en":
        user_message = user_content + "\nReply in English. There should be no other language. say what crops are recomended"
    else:
        user_message = user_content + "\nதமிழில் பதில் சொல்லுங்கள். வேறு எந்த மொழியும் இருக்கக்கூடாது. எந்த பயிர்கள் மட்டும் பரிந்துரைக்கப்படுகின்றன என்று சொல்லுங்கள்!"

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": user_message}
    ]

    message = call_gemini_api(messages)

    return jsonify({
        "response": message,
        "audio": text_to_speech(message)
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt_initial = [{"role": "system", "content": PROMPT}]
    chat_history = data.get("chatHistory", [])
    messages = prompt_initial + chat_history

    message = call_gemini_api(messages)

    return jsonify({
        "response": message,
        "audio": text_to_speech(message)
    })

@app.route("/api/health")
def health():
    return "ok", 200

if __name__ == '__main__':
    app.run(debug=True)