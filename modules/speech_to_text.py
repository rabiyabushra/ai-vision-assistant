"""
speech_to_text.py - Voice Command Recognition & Keyboard Fallback Module
Course: B.E. CSE Mini-Project — Deep Learning
Assigned to: Member B (Interaction & Systems)

Handles voice command recognition using SpeechRecognition (if hardware/microphones
are present) with a seamless, robust fallback to keyboard command parsing.
"""

from typing import Optional, Tuple

# Optional SpeechRecognition import
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    sr = None
    HAS_SPEECH_RECOGNITION = False


class SpeechRecognizer:
    """
    Voice Command and Keyboard Fallback Processor.
    Supports voice input listening via microphone and keyword parsing for UI control.
    """

    # Keyword mappings to standardized system actions
    COMMAND_KEYWORDS = {
        "start": ["start", "launch", "begin", "open camera", "start camera"],
        "stop": ["stop", "pause", "close camera", "stop camera", "quit", "exit"],
        "detect": ["detect", "scan", "what is this", "objects", "look"],
        "read_text": ["read text", "read text on this sign", "read sign", "read page", "ocr", "read"],
        "describe_scene": ["describe scene", "describe", "scene overview", "what is around me", "environment"],
        "toggle_speech": ["toggle speech", "mute", "unmute", "speech on", "speech off", "voice control"]
    }

    def __init__(self, language: str = "en-US"):
        """
        Initialize SpeechRecognizer.

        Args:
            language: Speech recognition language code (default 'en-US').
        """
        self.language = language
        self.recognizer = sr.Recognizer() if HAS_SPEECH_RECOGNITION else None
        self.stt_available = False
        self._check_hardware()

    def _check_hardware(self):
        """Verify if SpeechRecognition and a working microphone input exist."""
        if not HAS_SPEECH_RECOGNITION:
            self.stt_available = False
            return

        try:
            mics = sr.Microphone.list_microphone_names()
            self.stt_available = len(mics) > 0
        except Exception:
            self.stt_available = False

    def listen_voice_command(self, timeout: int = 5) -> Tuple[str, str]:
        """
        Listen to audio from microphone and attempt speech-to-text recognition.

        Args:
            timeout: Maximum seconds to listen for audio input.

        Returns:
            Tuple of (recognized_text, matched_action_command).
        """
        if not self.stt_available or self.recognizer is None:
            return "", ""

        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            
            recognized_text = self.recognizer.recognize_google(audio, language=self.language).lower()
            command = self.parse_command(recognized_text)
            return recognized_text, command
        except Exception as err:
            # Handles sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError, etc.
            return "", ""

    def parse_command(self, text_input: str) -> str:
        """
        Parse raw input string (voice phrase or keyboard string) into standardized action key.

        Args:
            text_input: Input string.

        Returns:
            str: Standardized action key ('start', 'stop', 'detect', 'read_text', 'describe_scene', 'toggle_speech', or '')
        """
        if not text_input or not text_input.strip():
            return ""

        import re
        query = text_input.strip().lower()

        for action, phrases in self.COMMAND_KEYWORDS.items():
            for phrase in phrases:
                pattern = r"\b" + re.escape(phrase) + r"\b"
                if re.search(pattern, query):
                    return action

        return ""

    def get_status_description(self) -> str:
        """Return human-readable status description of STT capability."""
        if self.stt_available:
            return "[Voice Mode] Microphone active (Voice commands supported)"
        else:
            return "[Keyboard Mode] Keyboard fallback mode active (Type or click command buttons)"


if __name__ == "__main__":
    print("=" * 65)
    print("AI Vision Assistant - Voice Command & Keyboard Fallback Test (Phase 8)")
    print("=" * 65)

    sp_rec = SpeechRecognizer()
    print(f"STT Hardware Status: {sp_rec.get_status_description()}")

    test_inputs = [
        "Please read text on this sign",
        "Describe scene around me",
        "Start camera feed",
        "Stop",
        "Detect objects in front",
        "Unmute speech",
        "Random unmapped text"
    ]

    print("\nTesting Command Parser across sample input phrases:")
    for query in test_inputs:
        cmd = sp_rec.parse_command(query)
        print(f" Input: \"{query}\" -> Action: '{cmd if cmd else 'NONE'}'")

    print("=" * 65)
