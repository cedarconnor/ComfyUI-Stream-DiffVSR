"""
Model loader node for Stream-DiffVSR.
"""

import torch
from typing import Tuple

from ..stream_diffvsr.models.loader import load_pipeline, ModelLoader
from ..stream_diffvsr.utils.device_utils import get_device, get_dtype


class StreamDiffVSR_Loader:
    """
    Load Stream-DiffVSR model components.

    This node initializes all model components (ControlNet, U-Net, Temporal VAE,
    Flow Estimator) and returns a pipeline object ready for inference.
    
    Model loading priority:
    1. Local path (ComfyUI/models/StreamDiffVSR/v1/) - preferred
    2. HuggingFace Hub (auto-download) - fallback
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (
                    ["v1"],
                    {
                        "default": "v1",
                        "tooltip": "Model version to load",
                    },
                ),
                "device": (
                    ["cuda", "cpu", "auto"],
                    {
                        "default": "auto",
                        "tooltip": "Device for inference. 'auto' selects CUDA if available.",
                    },
                ),
                "dtype": (
                    ["float16", "bfloat16", "float32"],
                    {
                        "default": "float16",
                        "tooltip": "Model precision. float16 recommended for most GPUs.",
                    },
                ),
            },
            "optional": {
                "use_local_models": (
                    ["true", "false"],
                    {
                        "default": "true",
                        "tooltip": (
                            "Try to load from ComfyUI/models/StreamDiffVSR/ first. "
                            "If false, always downloads from HuggingFace."
                        ),
                    },
                ),
                "use_huggingface": (
                    ["true", "false"],
                    {
                        "default": "true",
                        "tooltip": (
                            "Fall back to HuggingFace auto-download if local not found."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STREAM_DIFFVSR_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffVSR"
    DESCRIPTION = "Load Stream-DiffVSR model for video super-resolution"

    def load_model(
        self,
        model_version: str,
        device: str,
        dtype: str,
        use_local_models: str = "true",
        use_huggingface: str = "true",
    ) -> Tuple:
        """
        Load all model components and create pipeline.

        Args:
            model_version: Version of model to load
            device: Target device
            dtype: Model dtype
            use_local_models: Try local loading first
            use_huggingface: Fall back to HuggingFace

        Returns:
            Tuple containing pipeline
        """
        # Get device and dtype
        target_device = get_device(device)
        target_dtype = get_dtype(dtype)

        print(f"[Stream-DiffVSR] Loading model version {model_version}")
        print(f"[Stream-DiffVSR] Device: {target_device}, dtype: {target_dtype}")

        # Convert string bools
        use_local = use_local_models.lower() == "true"
        use_hf = use_huggingface.lower() == "true"

        # Try to get local model path
        local_path = None
        if use_local:
            try:
                local_path = ModelLoader.get_model_path()
                print(f"[Stream-DiffVSR] Local model path: {local_path}")
            except Exception as e:
                print(f"[Stream-DiffVSR] Local path not found: {e}")
                local_path = None

        # Load pipeline
        pipeline = load_pipeline(
            model_path=local_path,
            version=model_version,
            device=str(target_device),
            dtype=target_dtype,
            use_local=use_local,
            use_huggingface=use_hf,
        )

        print("[Stream-DiffVSR] Model loaded successfully!")

        return (pipeline,)
