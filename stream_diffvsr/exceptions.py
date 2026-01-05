"""
Custom exceptions for Stream-DiffVSR.

Provides informative error messages with actionable suggestions.
"""


class StreamDiffVSRError(Exception):
    """Base exception for Stream-DiffVSR errors."""

    pass


class ModelNotFoundError(StreamDiffVSRError):
    """Raised when model files are missing."""

    def __init__(self, component: str, expected_path: str):
        self.component = component
        self.expected_path = expected_path
        super().__init__(
            f"Model component '{component}' not found.\n"
            f"Expected location: {expected_path}\n\n"
            f"Please download models from:\n"
            f"  https://huggingface.co/Jamichsu/Stream-DiffVSR\n\n"
            f"And place them in:\n"
            f"  ComfyUI/models/StreamDiffVSR/"
        )


class ModelLoadError(StreamDiffVSRError):
    """Raised when model loading fails."""

    def __init__(self, component: str, path: str, reason: str):
        self.component = component
        self.path = path
        self.reason = reason
        super().__init__(
            f"Failed to load model component '{component}'.\n"
            f"Path: {path}\n"
            f"Reason: {reason}"
        )


class IncompatibleInputError(StreamDiffVSRError):
    """Raised when input dimensions are incompatible."""

    def __init__(self, expected: str, got: str, suggestion: str = ""):
        self.expected = expected
        self.got = got
        msg = (
            f"Incompatible input dimensions.\n"
            f"Expected: {expected}\n"
            f"Got: {got}"
        )
        if suggestion:
            msg += f"\nSuggestion: {suggestion}"
        super().__init__(msg)


class StateError(StreamDiffVSRError):
    """Raised when state management fails."""

    def __init__(self, message: str):
        super().__init__(f"State error: {message}")


class VRAMError(StreamDiffVSRError):
    """Raised when VRAM is insufficient."""

    def __init__(self, required_mb: int, available_mb: int, suggestion: str = ""):
        self.required_mb = required_mb
        self.available_mb = available_mb
        msg = (
            f"Insufficient VRAM.\n"
            f"Required: ~{required_mb} MB\n"
            f"Available: {available_mb} MB"
        )
        if suggestion:
            msg += f"\nSuggestion: {suggestion}"
        else:
            msg += "\nSuggestions:\n"
            msg += "  - Reduce frames_per_batch (for UpscaleVideo node)\n"
            msg += "  - Reduce batch size (for Upscale node)\n"
            msg += "  - Reduce input resolution\n"
            msg += "  - Use float16 instead of float32"
        super().__init__(msg)


class ConfigurationError(StreamDiffVSRError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")
