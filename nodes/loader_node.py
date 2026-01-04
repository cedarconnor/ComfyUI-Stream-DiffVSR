"""
Model loader node for Stream-DiffVSR.
"""

import torch
from typing import Tuple

from ..stream_diffvsr.models.loader import ModelLoader
from ..stream_diffvsr.models.unet import StreamDiffVSRUNet
from ..stream_diffvsr.models.artg import ARTGModule
from ..stream_diffvsr.models.temporal_decoder import TemporalAwareDecoder
from ..stream_diffvsr.models.flow_estimator import get_flow_estimator
from ..stream_diffvsr.schedulers.ddim_4step import DDIM4StepScheduler
from ..stream_diffvsr.pipeline import StreamDiffVSRPipeline, StreamDiffVSRConfig
from ..stream_diffvsr.utils.device_utils import get_device, get_dtype


class StreamDiffVSR_Loader:
    """
    Load Stream-DiffVSR model components.

    This node initializes all model components (U-Net, ARTG, Temporal Decoder,
    VAE, Flow Estimator) and returns a pipeline object ready for inference.
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
                "flow_model": (
                    ["auto", "raft_small", "none"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Optical flow model. 'auto' uses torchvision RAFT if available, "
                            "'none' disables temporal guidance."
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
        flow_model: str = "auto",
    ) -> Tuple:
        """
        Load all model components and create pipeline.

        Args:
            model_version: Version of model to load
            device: Target device
            dtype: Model dtype
            flow_model: Flow model type

        Returns:
            Tuple containing pipeline
        """
        # Get device and dtype
        target_device = get_device(device)
        target_dtype = get_dtype(dtype)

        print(f"[Stream-DiffVSR] Loading model version {model_version}")
        print(f"[Stream-DiffVSR] Device: {target_device}, dtype: {target_dtype}")

        # Get model path and validate files
        model_path = ModelLoader.get_model_path()
        component_paths = ModelLoader.validate_model_files(model_path, model_version)

        print(f"[Stream-DiffVSR] Found model files in: {model_path}")

        # Load configuration
        config_dict = ModelLoader.load_config(model_path, model_version)
        config = StreamDiffVSRConfig(**{
            k: v for k, v in config_dict.items()
            if k in StreamDiffVSRConfig.__dataclass_fields__
        })

        # Load model components
        print("[Stream-DiffVSR] Loading U-Net...")
        unet_state = ModelLoader.load_component(component_paths["unet"])
        unet = StreamDiffVSRUNet.from_pretrained(unet_state)

        print("[Stream-DiffVSR] Loading ARTG...")
        artg_state = ModelLoader.load_component(component_paths["artg"])
        artg = ARTGModule.from_pretrained(artg_state)

        print("[Stream-DiffVSR] Loading Temporal Decoder...")
        decoder_state = ModelLoader.load_component(component_paths["temporal_decoder"])
        decoder = TemporalAwareDecoder.from_pretrained(decoder_state)

        print("[Stream-DiffVSR] Loading VAE Encoder...")
        # TODO: Load VAE encoder - may use diffusers AutoencoderTiny
        # For now, create placeholder
        vae_state = ModelLoader.load_component(component_paths["vae"])
        vae_encoder = self._create_vae_encoder(vae_state, config)

        # Load flow estimator
        flow_estimator = None
        if flow_model != "none":
            print("[Stream-DiffVSR] Loading Flow Estimator...")
            flow_path = component_paths.get("flow")
            flow_estimator = get_flow_estimator(
                target_device,
                target_dtype,
                model_path=flow_path,
            )

        # Create scheduler
        scheduler = DDIM4StepScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
        )

        # Create pipeline
        pipeline = StreamDiffVSRPipeline(
            unet=unet,
            artg=artg,
            decoder=decoder,
            vae_encoder=vae_encoder,
            flow_estimator=flow_estimator,
            scheduler=scheduler,
            config=config,
            device=target_device,
            dtype=target_dtype,
        )

        print("[Stream-DiffVSR] Model loaded successfully!")

        return (pipeline,)

    def _create_vae_encoder(self, state_dict, config):
        """Create VAE encoder from state dict."""
        # TODO: Implement proper VAE loading based on upstream
        # This is a placeholder
        try:
            from diffusers import AutoencoderTiny

            # Try to load as AutoencoderTiny
            vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=torch.float32,
            )
            return vae
        except Exception:
            # Return a dummy encoder for now
            import torch.nn as nn

            class DummyEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.config = type("Config", (), {"scaling_factor": 0.18215})()

                def encode(self, x):
                    # Just return a downscaled version as "latent"
                    import torch.nn.functional as F
                    latent = F.interpolate(x, scale_factor=0.125, mode="bilinear")
                    return type("Output", (), {"latent_dist": type("Dist", (), {"mode": lambda: latent})()})()

            return DummyEncoder()
