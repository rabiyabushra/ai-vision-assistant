"""
text_to_speech.py - Text-to-Speech (TTS) Synthesis Module
Course: B.E. CSE Mini-Project — Deep Learning
Assigned to: Member B (Interaction & Systems)

Provides cross-platform offline text-to-speech synthesis using pyttsx3.
Runs speech synthesis in a background worker thread to prevent blocking
the main Streamlit UI rendering thread and avoid Windows COM concurrency issues.
"""

import queue
import threading
import time
from typing import Optional
import pyttsx3


class Speaker:
    """
    Non-blocking Offline Text-to-Speech Engine.
    Uses a background queue to synthesize spoken text without blocking the main app.
    """

    def __init__(self, rate: int = 160, enabled: bool = True):
        """
        Initialize TTS speaker settings.

        Args:
            rate: Speech speed rate in words per minute (default 160).
            enabled: Initial toggle state for speech output.
        """
        self.rate = rate
        self.enabled = enabled
        self._speech_queue: queue.Queue = queue.Queue()
        self._running = True
        self._is_speaking = False
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def _process_queue(self):
        """Background thread loop managing the pyttsx3 engine lifecycle."""
        # Initialize pyttsx3 engine within the worker thread context
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        except Exception as e:
            print(f"[!] Warning: Could not initialize pyttsx3 engine: {e}")
            engine = None

        while self._running:
            try:
                text = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not text or not self.enabled or engine is None:
                self._speech_queue.task_done()
                continue

            self._is_speaking = True
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as err:
                print(f"[!] TTS synthesis error: {err}")
                # Re-initialize engine if COM state is corrupted
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self.rate)
                except Exception:
                    pass
            finally:
                self._is_speaking = False
                self._speech_queue.task_done()

    def speak(self, text: str, clear_queue: bool = False):
        """
        Queue text for spoken output.

        Args:
            text: Sentence or message string to speak.
            clear_queue: If True, purges pending items before adding the new text.
        """
        if not text or not text.strip() or not self.enabled:
            return

        if clear_queue:
            self.stop_speech()

        self._speech_queue.put(text.strip())

    def stop_speech(self):
        """Purge all pending text messages from the speech queue."""
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except queue.Empty:
                break

    def set_enabled(self, enabled: bool):
        """Enable or disable spoken audio output."""
        self.enabled = enabled
        if not enabled:
            self.stop_speech()

    def is_enabled(self) -> bool:
        """Return current speech enabled status."""
        return self.enabled

    def is_speaking(self) -> bool:
        """Return True if currently speaking text."""
        return self._is_speaking

    def set_rate(self, rate: int):
        """Update speech rate."""
        self.rate = rate

    def close(self):
        """Shut down worker thread cleanly."""
        self._running = False
        self.stop_speech()


if __name__ == "__main__":
    print("=" * 60)
    print("AI Vision Assistant - Text-To-Speech (TTS) Test (Phase 4)")
    print("=" * 60)

    speaker = Speaker(rate=165, enabled=True)
    print("Enqueuing test voice response...")
    speaker.speak("AI Vision Assistant ready. Text to speech module operating correctly.")

    # Wait for queue to finish speaking
    t0 = time.time()
    while speaker.is_speaking() or not speaker._speech_queue.empty():
        if time.time() - t0 > 10.0:
            print("[!] Timeout waiting for speech.")
            break
        time.sleep(0.1)

    print("[OK] Speech completed.")
    speaker.close()
    print("=" * 60)
