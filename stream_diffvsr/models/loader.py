"""
Model loader for Stream-DiffVSR components.

Handles discovery and loading of model files from ComfyUI's model directory.
"""

import os
from typing import Dict, List, Optional, Tuple
import torch
from safetensors.torch import load_file

from ..exceptions import ModelNotFoundError, ModelLoadError


class ModelLoader:
    """
    Handles loading Stream-DiffVSR model components.

    Searches for models in:
    1. ComfyUI/models/StreamDiffVSR/{version}/
    2. Custom paths specified by user
    """

    REQUIRED_COMPONENTS = [
        "unet",
        "artg",
        "temporal_decoder",
        "vae",
    ]

    OPTIONAL_COMPONENTS = [
        "flow",  # Can fall back to torchvision RAFT
    ]

    SUPPORTED_EXTENSIONS = [".safetensors", ".pth", ".ckpt", ".bin"]

    @classmethod
    def get_model_path(cls, custom_path: Optional[str] = None) -> str:
        """
        Get the StreamDiffVSR model directory.

        Args:
            custom_path: Optional custom path to use

        Returns:
            Path to model directory

        Raises:
            ModelNotFoundError: If directory doesn't exist
        """
        if custom_path and os.path.exists(custom_path):
            return custom_path

        # Try to import ComfyUI's folder_paths
        try:
            import folder_paths

            # Register custom model path if not exists
            if "StreamDiffVSR" not in folder_paths.folder_names_and_paths:
                model_path = os.path.join(folder_paths.models_dir, "StreamDiffVSR")
                folder_paths.folder_names_and_paths["StreamDiffVSR"] = (
                    [model_path],
                    set(cls.SUPPORTED_EXTENSIONS),
                )

            paths = folder_paths.get_folder_paths("StreamDiffVSR")
            if paths and os.path.exists(paths[0]):
                return paths[0]

            # Default path
            default_path = os.path.join(folder_paths.models_dir, "StreamDiffVSR")
            if os.path.exists(default_path):
                return default_path

            raise ModelNotFoundError(
                "StreamDiffVSR",
                default_path,
            )

        except ImportError:
            # Not running in ComfyUI context
            # Try common locations
            common_paths = [
                os.path.expanduser("~/.cache/comfyui/models/StreamDiffVSR"),
                os.path.expanduser("~/ComfyUI/models/StreamDiffVSR"),
                "./models/StreamDiffVSR",
            ]

            for path in common_paths:
                if os.path.exists(path):
                    return path

            raise ModelNotFoundError(
                "StreamDiffVSR",
                "ComfyUI/models/StreamDiffVSR/",
            )

    @classmethod
    def find_model_file(cls, directory: str) -> Optional[str]:
        """
        Find a model file in a directory.

        Args:
            directory: Directory to search

        Returns:
            Path to model file, or None if not found
        """
        if not os.path.isdir(directory):
            return None

        for ext in cls.SUPPORTED_EXTENSIONS:
            for filename in os.listdir(directory):
                if filename.endswith(ext):
                    return os.path.join(directory, filename)

        return None

    @classmethod
    def validate_model_files(
        cls,
        model_path: str,
        version: str = "v1",
    ) -> Dict[str, str]:
        """
        Validate that all required model files exist.

        Args:
            model_path: Base path to StreamDiffVSR models
            version: Model version (e.g., "v1")

        Returns:
            Mapping of component name to file path

        Raises:
            ModelNotFoundError: If required components are missing
        """
        version_path = os.path.join(model_path, version)

        # Check if version directory exists
        if not os.path.exists(version_path):
            # Maybe files are directly in model_path
            if cls.find_model_file(os.path.join(model_path, "unet")):
                version_path = model_path
            else:
                raise ModelNotFoundError(
                    f"version {version}",
                    version_path,
                )

        component_paths = {}
        missing = []

        for component in cls.REQUIRED_COMPONENTS:
            component_dir = os.path.join(version_path, component)
            model_file = cls.find_model_file(component_dir)

            if model_file:
                component_paths[component] = model_file
            else:
                missing.append(component)

        if missing:
            raise ModelNotFoundError(
                f"components: {', '.join(missing)}",
                version_path,
            )

        # Check optional components
        for component in cls.OPTIONAL_COMPONENTS:
            component_dir = os.path.join(version_path, component)
            model_file = cls.find_model_file(component_dir)
            if model_file:
                component_paths[component] = model_file

        return component_paths

    @classmethod
    def load_component(
        cls,
        path: str,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Load a single model component.

        Args:
            path: Path to model file
            device: Device to load to

        Returns:
            State dict

        Raises:
            ModelLoadError: If loading fails
        """
        try:
            if path.endswith(".safetensors"):
                return load_file(path, device=device)
            else:
                return torch.load(path, map_location=device, weights_only=True)
        except Exception as e:
            raise ModelLoadError(
                os.path.basename(os.path.dirname(path)),
                path,
                str(e),
            )

    @classmethod
    def load_config(cls, model_path: str, version: str = "v1") -> Dict:
        """
        Load model configuration.

        Args:
            model_path: Base path to StreamDiffVSR models
            version: Model version

        Returns:
            Configuration dictionary
        """
        import json

        # Try version-specific config
        config_path = os.path.join(model_path, version, "config.json")
        if not os.path.exists(config_path):
            # Try base config
            config_path = os.path.join(model_path, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)

        # Return default config
        return {
            "scale_factor": 4,
            "latent_channels": 4,
            "latent_scale": 8,
            "num_inference_steps": 4,
            "vae_scaling_factor": 0.18215,  # Will be overridden by VAE config
        }
