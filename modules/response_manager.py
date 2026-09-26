import time

class ResponseManager:
    def __init__(self, cooldown_seconds=4):
        self.cooldown_seconds = cooldown_seconds
        self.last_response = ""
        self.last_time = 0.0

    def object_response(self, detections):
        if not detections:
            return ""

        labels = []
        for d in detections:
            if d["label"] not in labels:
                labels.append(d["label"])

        response = self._make_response(labels)
        now = time.time()

        if response == self.last_response and now - self.last_time < self.cooldown_seconds:
            return ""

        self.last_response = response
        self.last_time = now
        return response

    @staticmethod
    def _make_response(labels):
        if len(labels) == 1:
            return f"{labels[0]} detected."
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]} detected."
        return ", ".join(labels[:-1]) + f", and {labels[-1]} detected."
