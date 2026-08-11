import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { textToGif } from "../api";
import TextToGif from "./TextToGif";

vi.mock("../api", () => ({ textToGif: vi.fn() }));

describe("TextToGif", () => {
  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => "blob:gif");
    URL.revokeObjectURL = vi.fn();
  });

  it("submits normalized text and displays the generated GIF", async () => {
    textToGif.mockResolvedValue(new Blob(["GIF89a"], { type: "image/gif" }));
    const { unmount } = render(<TextToGif />);

    fireEvent.change(screen.getByLabelText("Text to fingerspell"), {
      target: { value: "  Hi  " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate GIF" }));

    await waitFor(() => expect(textToGif).toHaveBeenCalledWith("Hi"));
    expect(screen.getByRole("img")).toHaveAttribute("src", "blob:gif");
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:gif");
  });

  it("shows backend errors", async () => {
    textToGif.mockRejectedValue(new Error("Sign images unavailable"));
    render(<TextToGif />);
    fireEvent.change(screen.getByLabelText("Text to fingerspell"), {
      target: { value: "hello" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate GIF" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("unavailable"));
  });
});
