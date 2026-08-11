import { afterEach, describe, expect, it, vi } from "vitest";
import { errorText, getApplicationStatus, textToGif } from "./api";

describe("API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("returns application status JSON", async () => {
    const payload = { status: "ready", capabilities: {} };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(payload))));

    await expect(getApplicationStatus()).resolves.toEqual(payload);
    expect(fetch).toHaveBeenCalledWith("/api/status");
  });

  it("returns a GIF Blob", async () => {
    const blob = new Blob(["GIF89a"], { type: "image/gif" });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, blob: async () => blob }));

    await expect(textToGif("hello")).resolves.toHaveProperty("type", "image/gif");
  });

  it("extracts FastAPI error details", async () => {
    const response = new Response(JSON.stringify({ detail: "missing assets" }), { status: 503 });
    await expect(errorText(response)).resolves.toBe("missing assets");
  });
});
