"""
Model loader for Stream-DiffVSR components.

Supports loading from:
1. Local path (ComfyUI/models/StreamDiffVSR/v1/) - MANUAL DOWNLOAD
2. HuggingFace Hub (Jamichsu/Stream-DiffVSR) - AUTO DOWNLOAD (fallback)
"""

import os
from typing import Dict, List, Optional, Tuple
import torch
from safetensors.torch import load_file

from ..exceptions import ModelNotFoundError, ModelLoadError


# HuggingFace model ID for auto-download
HUGGINGFACE_MODEL_ID = "Jamichsu/Stream-DiffVSR"


class ModelLoader:
    """
    Handles loading Stream-DiffVSR model components.

    Loading priority:
    1. Local path (ComfyUI/models/StreamDiffVSR/{version}/) - preferred
    2. HuggingFace Hub (auto-download) - fallback
    
    Local directory structure:
    ```
    ComfyUI/models/StreamDiffVSR/v1/
    ├── controlnet/              # ControlNet for temporal guidance
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors
    ├── unet/                    # U-Net for denoising
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors
    ├── vae/                     # Temporal VAE with TPM
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors
    └── scheduler/               # DDIM scheduler config
        └── scheduler_config.json
    ```
    """

    REQUIRED_COMPONENTS = [
        "controlnet",
        "unet",
        "vae",
    ]

    OPTIONAL_COMPONENTS = [
        "scheduler",
        "text_encoder",
        "tokenizer",
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
    def has_local_models(
        cls,
        model_path: str,
        version: str = "v1",
    ) -> bool:
        """
        Check if local model files exist.

        Args:
            model_path: Base path to StreamDiffVSR models
            version: Model version (e.g., "v1")

        Returns:
            True if all required components exist locally
        """
        version_path = os.path.join(model_path, version)
        
        if not os.path.exists(version_path):
            # Check if files are directly in model_path (flat structure)
            version_path = model_path

        for component in cls.REQUIRED_COMPONENTS:
            component_dir = os.path.join(version_path, component)
            if not os.path.isdir(component_dir):
                return False
            # Check for config.json (diffusers format)
            if not os.path.exists(os.path.join(component_dir, "config.json")):
                return False

        return True

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
            Mapping of component name to directory path

        Raises:
            ModelNotFoundError: If required components are missing
        """
        version_path = os.path.join(model_path, version)

        # Check if version directory exists
        if not os.path.exists(version_path):
            # Maybe files are directly in model_path
            if os.path.isdir(os.path.join(model_path, "unet")):
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
            if os.path.isdir(component_dir):
                component_paths[component] = component_dir
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
            if os.path.isdir(component_dir):
                component_paths[component] = component_dir

        return component_paths

    @classmethod
    def load_component(
        cls,
        path: str,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Load a single model component state dict.

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
            "latent_scale": 4,
            "num_inference_steps": 4,
            "vae_scaling_factor": 1.0,  # AutoEncoderTiny uses 1.0
        }


def load_pipeline(
    model_path: Optional[str] = None,
    version: str = "v1",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_local: bool = True,
    use_huggingface: bool = True,
):
    """
    Load complete Stream-DiffVSR pipeline.

    Loading priority:
    1. Local models (if use_local=True and models exist)
    2. HuggingFace (if use_huggingface=True)

    Args:
        model_path: Custom local model path (optional)
        version: Model version for local loading
        device: Target device
        dtype: Model dtype
        use_local: Whether to try local loading first
        use_huggingface: Whether to fall back to HuggingFace

    Returns:
        StreamDiffVSRPipeline instance
    """
    from .controlnet import TemporalControlNet
    from .unet import StreamDiffVSRUNet
    from .temporal_vae import TemporalVAE
    from .flow_estimator import get_flow_estimator
    from ..pipeline import StreamDiffVSRPipeline, StreamDiffVSRConfig
    
    try:
        from diffusers import DDIMScheduler
    except ImportError:
        raise ImportError("diffusers is required. Install with: pip install diffusers>=0.25.0")

    # Determine loading source
    use_local_path = False
    local_path = None
    
    if use_local:
        try:
            if model_path:
                local_path = model_path
            else:
                local_path = ModelLoader.get_model_path()
            
            if ModelLoader.has_local_models(local_path, version):
                use_local_path = True
                print(f"[Stream-DiffVSR] Loading from local: {local_path}/{version}")
            else:
                print(f"[Stream-DiffVSR] Local models not found at {local_path}/{version}")
        except ModelNotFoundError:
            print("[Stream-DiffVSR] Local model directory not found")

    if use_local_path:
        # Load from local paths
        version_path = os.path.join(local_path, version)
        if not os.path.exists(version_path):
            version_path = local_path
        
        controlnet = TemporalControlNet.from_local(
            os.path.join(version_path, "controlnet"),
            torch_dtype=dtype,
        )
        unet = StreamDiffVSRUNet.from_local(
            os.path.join(version_path, "unet"),
            torch_dtype=dtype,
        )
        vae = TemporalVAE.from_local(
            os.path.join(version_path, "vae"),
            torch_dtype=dtype,
        )
        scheduler = DDIMScheduler.from_pretrained(
            version_path,
            subfolder="scheduler",
        )
        
    elif use_huggingface:
        # Load from HuggingFace
        print(f"[Stream-DiffVSR] Loading from HuggingFace: {HUGGINGFACE_MODEL_ID}")
        
        controlnet = TemporalControlNet.from_pretrained(
            HUGGINGFACE_MODEL_ID,
            subfolder="controlnet",
            torch_dtype=dtype,
        )
        unet = StreamDiffVSRUNet.from_pretrained(
            HUGGINGFACE_MODEL_ID,
            subfolder="unet",
            torch_dtype=dtype,
        )
        vae = TemporalVAE.from_pretrained(
            HUGGINGFACE_MODEL_ID,
            subfolder="vae",
            torch_dtype=dtype,
        )
        scheduler = DDIMScheduler.from_pretrained(
            HUGGINGFACE_MODEL_ID,
            subfolder="scheduler",
        )
    else:
        raise ModelNotFoundError(
            "Stream-DiffVSR models",
            "No loading source available. Enable use_local or use_huggingface.",
        )

    # Load flow estimator (always from torchvision)
    flow_estimator = get_flow_estimator(device=device)

    # Create pipeline config
    config = StreamDiffVSRConfig(
        scale_factor=4,
        latent_channels=4,
        latent_scale=4,
        num_inference_steps=4,
        vae_scaling_factor=1.0,
    )

    # Create pipeline
    pipeline = StreamDiffVSRPipeline(
        unet=unet.to(device),
        controlnet=controlnet.to(device),
        vae=vae.to(device),
        scheduler=scheduler,
        flow_estimator=flow_estimator,
        config=config,
        device=torch.device(device),
        dtype=dtype,
    )

    return pipeline
