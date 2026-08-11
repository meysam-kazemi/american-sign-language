import { useCallback, useEffect, useRef, useState } from "react";

// Records microphone audio via MediaRecorder and returns the captured Blob.
export function useRecorder() {
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState("");

  const start = useCallback(async () => {
    setError("");
    try {
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
        throw new Error("Audio recording is not supported by this browser");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
      return true;
    } catch (err) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setError(err.message || "Could not access microphone");
      setRecording(false);
      return false;
    }
  }, []);

  const stop = useCallback(
    () =>
      new Promise((resolve) => {
        const recorder = recorderRef.current;
        if (!recorder) return resolve(null);
        recorder.onstop = () => {
          recorder.stream.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
          recorderRef.current = null;
          setRecording(false);
          resolve(new Blob(chunksRef.current, { type: recorder.mimeType }));
        };
        recorder.onerror = () => {
          recorder.stream.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
          recorderRef.current = null;
          setRecording(false);
          setError("The browser could not finish the audio recording");
          resolve(null);
        };
        recorder.stop();
      }),
    []
  );

  useEffect(
    () => () => {
      const recorder = recorderRef.current;
      if (recorder?.state === "recording") recorder.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    },
    []
  );

  return { recording, error, start, stop };
}
