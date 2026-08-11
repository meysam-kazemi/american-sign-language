import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useWebcam } from "./useWebcam";

describe("useWebcam", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("returns false and exposes permission errors", async () => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error("Permission denied")) },
    });
    const { result } = renderHook(() => useWebcam());

    await act(async () => expect(await result.current.start()).toBe(false));
    expect(result.current.active).toBe(false);
    expect(result.current.error).toBe("Permission denied");
  });
});
