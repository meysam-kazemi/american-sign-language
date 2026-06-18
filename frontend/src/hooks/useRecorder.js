import { useCallback, useRef, useState } from "react";

// Records microphone audio via MediaRecorder and returns the captured Blob.
export function useRecorder() {
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState("");

  const start = useCallback(async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      setError(err.message || "Could not access microphone");
    }
  }, []);

  const stop = useCallback(
    () =>
      new Promise((resolve) => {
        const recorder = recorderRef.current;
        if (!recorder) return resolve(null);
        recorder.onstop = () => {
          recorder.stream.getTracks().forEach((t) => t.stop());
          setRecording(false);
          resolve(new Blob(chunksRef.current, { type: recorder.mimeType }));
        };
        recorder.stop();
      }),
    []
  );

  return { recording, error, start, stop };
}
