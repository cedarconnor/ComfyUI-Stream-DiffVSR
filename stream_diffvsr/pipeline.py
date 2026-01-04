"""
Main inference pipeline for Stream-DiffVSR.

This pipeline owns the entire denoising loop because ARTG requires
mid-network feature injection - we cannot use ComfyUI's standard sampler.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

from .state import StreamDiffVSRState
from .models.unet import StreamDiffVSRUNet
from .models.artg import ARTGModule
from .models.temporal_decoder import TemporalAwareDecoder
from .models.flow_estimator import FlowEstimator, ZeroFlowEstimator
from .schedulers.ddim_4step import DDIM4StepScheduler
from .utils.image_utils import bhwc_to_bchw, bchw_to_bhwc, normalize_to_neg1_1, denormalize_from_neg1_1
from .utils.flow_utils import warp_image, upscale_flow


@dataclass
class StreamDiffVSRConfig:
    """Configuration for Stream-DiffVSR pipeline."""

    scale_factor: int = 4
    latent_channels: int = 4
    latent_scale: int = 8  # Spatial downscale in latent space
    num_inference_steps: int = 4
    vae_scaling_factor: float = 0.18215  # Will be overridden by VAE config


class StreamDiffVSRPipeline:
    """
    Stream-DiffVSR inference pipeline.

    Implements the auto-regressive diffusion framework for video
    super-resolution with temporal guidance from previous frames.

    CRITICAL: This pipeline owns the denoising loop. We cannot use
    ComfyUI's KSampler because ARTG injects features mid-network.
    """

    def __init__(
        self,
        unet: StreamDiffVSRUNet,
        artg: ARTGModule,
        decoder: TemporalAwareDecoder,
        vae_encoder: torch.nn.Module,
        flow_estimator: Optional[FlowEstimator],
        scheduler: DDIM4StepScheduler,
        config: StreamDiffVSRConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize pipeline.

        Args:
            unet: Distilled U-Net model
            artg: ARTG temporal guidance module
            decoder: Temporal-aware VAE decoder
            vae_encoder: VAE encoder for LQ frames
            flow_estimator: Optical flow estimator (or None)
            scheduler: DDIM scheduler
            config: Pipeline configuration
            device: Target device
            dtype: Model dtype
        """
        self.unet = unet
        self.artg = artg
        self.decoder = decoder
        self.vae_encoder = vae_encoder
        self.flow_estimator = flow_estimator or ZeroFlowEstimator()
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.dtype = dtype

        # Move all models to device
        self._to_device()

    def _to_device(self):
        """Move all components to target device and dtype."""
        self.unet = self.unet.to(self.device, self.dtype)
        self.artg = self.artg.to(self.device, self.dtype)
        self.decoder = self.decoder.to(self.device, self.dtype)
        self.vae_encoder = self.vae_encoder.to(self.device, self.dtype)
        if self.flow_estimator is not None:
            self.flow_estimator = self.flow_estimator.to(self.device)

    @torch.inference_mode()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            image: (B, H, W, C) tensor in [0, 1] range (ComfyUI format)

        Returns:
            Latent tensor (B, C, h, w) where h = H/8, w = W/8
        """
        # ComfyUI format (BHWC) -> model format (BCHW)
        x = bhwc_to_bchw(image).to(self.device, self.dtype)

        # Normalize to [-1, 1]
        x = normalize_to_neg1_1(x)

        # Encode - handle different VAE output types
        encoder_output = self.vae_encoder.encode(x)

        # diffusers VAEs return distribution objects, not raw tensors
        if hasattr(encoder_output, "latent_dist"):
            # Standard diffusers VAE - use mode for deterministic encoding
            latent = encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            # Some VAEs return .latents directly
            latent = encoder_output.latents
        else:
            # Raw tensor output (e.g., AutoEncoderTiny)
            latent = encoder_output

        # Apply VAE-specific scaling factor from config
        # DO NOT hardcode 0.18215 - use config value
        scaling_factor = getattr(
            self.vae_encoder.config,
            "scaling_factor",
            self.config.vae_scaling_factor,
        )
        latent = latent * scaling_factor

        return latent

    @torch.inference_mode()
    def estimate_flow(
        self,
        current_lq: torch.Tensor,
        previous_lq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate optical flow from previous to current frame.

        Args:
            current_lq: Current LQ frame (B, C, H, W) in [0, 1]
            previous_lq: Previous LQ frame (B, C, H, W) in [0, 1]

        Returns:
            Flow field (B, 2, H, W)
        """
        return self.flow_estimator(previous_lq, current_lq)

    @torch.inference_mode()
    def process_frame(
        self,
        lq_frame: torch.Tensor,
        state: StreamDiffVSRState,
        num_inference_steps: int = 4,
        seed: int = 0,
    ) -> Tuple[torch.Tensor, StreamDiffVSRState]:
        """
        Process a single frame with auto-regressive temporal guidance.

        Args:
            lq_frame: Low-quality input frame (1, H, W, C) in [0, 1] BHWC
            state: Previous frame state
            num_inference_steps: Number of denoising steps
            seed: Random seed

        Returns:
            hq_frame: High-quality output frame (1, H*4, W*4, C) BHWC
            new_state: Updated state for next frame
        """
        # Setup
        generator = torch.Generator(device=self.device).manual_seed(seed)
        B, H, W, C = lq_frame.shape
        scale = self.config.scale_factor
        target_h, target_w = H * scale, W * scale

        # Convert to model format (BCHW, on device)
        lq_bchw = bhwc_to_bchw(lq_frame).to(self.device, self.dtype)

        # Encode LQ frame to latent
        z_lq = self.encode_image(lq_frame)

        # Handle temporal guidance
        warped_hq = None
        temporal_features = None

        if state.has_previous:
            # Previous frame data is available
            prev_lq = state.previous_lq.to(self.device, self.dtype)
            prev_hq = state.previous_hq.to(self.device, self.dtype)

            # Estimate flow from previous LQ to current LQ
            flow = self.estimate_flow(lq_bchw, prev_lq)

            # Upscale flow to HQ resolution
            flow_upscaled = upscale_flow(flow, scale)

            # Warp previous HQ using upscaled flow
            warped_hq = warp_image(prev_hq, flow_upscaled)

            # Get temporal features from ARTG
            temporal_features = self.artg.encode_temporal(warped_hq, z_lq)

        # Initialize noise for denoising
        latent_h = target_h // self.config.latent_scale
        latent_w = target_w // self.config.latent_scale

        noise = torch.randn(
            (B, self.config.latent_channels, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = noise * self.scheduler.init_noise_sigma

        # ===== DENOISING LOOP =====
        # This is why we own the loop - ARTG injects at each step
        for t in self.scheduler.timesteps:
            # U-Net prediction with ARTG temporal conditioning
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=z_lq,
                temporal_features=temporal_features,
            )

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # ===== TEMPORAL DECODE =====
        # Decode with temporal-aware decoder
        hq_bchw = self.decoder(
            latents,
            warped_previous=warped_hq,
            lq_features=z_lq,
        )

        # Clamp and convert to [0, 1]
        hq_bchw = torch.clamp(hq_bchw, -1, 1)
        hq_bchw = denormalize_from_neg1_1(hq_bchw)

        # Convert back to ComfyUI format (BHWC, CPU, float32)
        hq_frame = bchw_to_bhwc(hq_bchw).cpu().float()

        # Create new state (store as BCHW for efficiency)
        new_state = StreamDiffVSRState(
            previous_hq=hq_bchw.cpu().float(),  # BCHW
            previous_lq=lq_bchw.cpu().float(),  # BCHW
            frame_index=state.frame_index + 1,
        )

        return hq_frame, new_state

    @torch.inference_mode()
    def __call__(
        self,
        images: torch.Tensor,
        state: Optional[StreamDiffVSRState] = None,
        num_inference_steps: int = 4,
        seed: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, StreamDiffVSRState]:
        """
        Process a batch of frames.

        Batch dimension is treated as time - frames are processed
        sequentially with temporal guidance from previous frames.

        Args:
            images: Input frames (B, H, W, C) in [0, 1] range BHWC
                   B = number of frames (processed in order)
            state: Optional previous state for continuing a sequence
            num_inference_steps: Denoising steps per frame (default 4)
            seed: Random seed (incremented per frame)
            progress_callback: Optional callback(frame_idx, total_frames)

        Returns:
            hq_images: Upscaled frames (B, H*4, W*4, C) BHWC
            final_state: State after last frame
        """
        num_frames = images.shape[0]

        # Initialize state if not provided
        if state is None:
            state = StreamDiffVSRState()

        hq_frames = []

        for i in range(num_frames):
            # Extract single frame (keep batch dim)
            frame = images[i : i + 1]

            # Process with temporal guidance
            hq_frame, state = self.process_frame(
                frame,
                state,
                num_inference_steps=num_inference_steps,
                seed=seed + i,  # Different seed per frame
            )

            hq_frames.append(hq_frame)

            if progress_callback is not None:
                progress_callback(i + 1, num_frames)

        # Stack frames back to batch
        hq_images = torch.cat(hq_frames, dim=0)

        return hq_images, state
