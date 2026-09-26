import pyttsx3

class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)

    def speak(self, text):
        if not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()
