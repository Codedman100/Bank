from flask import Flask, render_template, request, jsonify
import pyttsx3
import threading

app = Flask(__name__)

# Initialize pyttsx3 engine
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
lock = threading.Lock()  # Lock to prevent multiple threads from calling TTS simultaneously

@app.route("/")
def main_menu():
    return render_template("main_menu.html")

@app.route("/speak", methods=["POST"])
def speak():
    try:
        data = request.json
        text = data.get("text", "").strip()
        voice = data.get("voice", "male")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Acquire the lock before using pyttsx3
        with lock:
            if voice == "male":
                tts_engine.setProperty("voice", voices[0].id)
            elif voice == "female":
                tts_engine.setProperty("voice", voices[1].id)

            tts_engine.say(text)
            tts_engine.runAndWait()

        return jsonify({"message": "Text spoken successfully!"})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
