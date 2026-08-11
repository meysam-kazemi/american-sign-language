"""Turn a noisy stream of per-frame sign predictions into stable text.

A prediction must be repeated for `stability` consecutive frames (above a
confidence threshold) before it is committed, which debounces flicker. The
special labels 'space' and 'nothing' are handled when building the sentence.
"""

from collections import deque


class StreamDecoder:
    def __init__(self, stability=8, min_confidence=0.6):
        if not isinstance(stability, int) or stability <= 0:
            raise ValueError("stability must be a positive integer")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.stability = stability
        self.min_confidence = min_confidence
        self.recent = deque(maxlen=stability)
        self.text = ""
        self._last_committed = None

    def update(self, label, confidence):
        """Feed one frame's prediction. Returns the current decoded text."""
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if label is None or confidence < self.min_confidence:
            self.recent.clear()
            self._last_committed = None
            return self.text

        label = label.lower()
        self.recent.append(label)
        if len(self.recent) == self.stability and len(set(self.recent)) == 1:
            stable = self.recent[0]
            if stable != self._last_committed:
                self._commit(stable)
                self._last_committed = stable
        return self.text

    def _commit(self, label):
        if label == "nothing":
            return
        if label == "space":
            self.text += " "
        elif label == "del":
            self.text = self.text[:-1]
        else:
            self.text += label.upper()

    def reset(self):
        self.recent.clear()
        self.text = ""
        self._last_committed = None
