import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { getApplicationStatus } from "./api";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal();
  return { ...original, getApplicationStatus: vi.fn() };
});

describe("App", () => {
  beforeEach(() => {
    getApplicationStatus.mockResolvedValue({ status: "ready", capabilities: {} });
  });

  it("reports backend readiness and switches translation modes", async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText("API: ready")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: "Text → GIF" }));
    expect(screen.getByRole("heading", { name: "Text → ASL GIF" })).toBeInTheDocument();
  });

  it("shows a useful message when the backend is offline", async () => {
    getApplicationStatus.mockRejectedValue(new Error("offline"));
    render(<App />);

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("backend is offline"));
  });
});
