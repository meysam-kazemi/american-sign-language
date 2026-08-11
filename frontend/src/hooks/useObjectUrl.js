import { useCallback, useEffect, useRef, useState } from "react";

/** Own an object URL and revoke it whenever it is replaced or unmounted. */
export function useObjectUrl() {
  const currentRef = useRef(null);
  const [url, setUrl] = useState(null);

  const clear = useCallback(() => {
    if (currentRef.current) URL.revokeObjectURL(currentRef.current);
    currentRef.current = null;
    setUrl(null);
  }, []);

  const setBlob = useCallback((blob) => {
    if (currentRef.current) URL.revokeObjectURL(currentRef.current);
    const nextUrl = blob ? URL.createObjectURL(blob) : null;
    currentRef.current = nextUrl;
    setUrl(nextUrl);
  }, []);

  useEffect(() => clear, [clear]);

  return { url, setBlob, clear };
}
