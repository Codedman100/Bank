from flask import Flask, render_template, request, jsonify
import pyttsx3
import threading

app = Flask(__name__)

# Initialize pyttsx3 engine
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
lock = threading.Lock()

@app.route("/")
def main_menu():
    return render_template("main_menu.html")

@app.route("/build-a-sentence", methods=["GET", "POST"])
def build_a_sentence():
    if request.method == "POST":
        try:
            data = request.json
            sentence = data.get("sentence", "").strip()
            voice = data.get("voice", "male")

            # Set the voice
            with lock:
                if voice == "male":
                    tts_engine.setProperty("voice", voices[0].id)
                elif voice == "female":
                    tts_engine.setProperty("voice", voices[1].id)

                # Speak the sentence
                tts_engine.say(sentence)
                tts_engine.runAndWait()

            return jsonify({"message": "Sentence spoken successfully!"})
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    return render_template("build_a_sentence.html")

@app.route("/make-a-custom-word")
def make_a_custom_word():
    return render_template("make_a_custom_word.html")

if __name__ == "__main__":
    app.run(debug=True)
