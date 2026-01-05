"""
Main inference pipeline for Stream-DiffVSR.

This pipeline owns the entire denoising loop because ControlNet temporal
guidance requires mid-network feature injection - we cannot use ComfyUI's
standard sampler.

Architecture:
- ControlNet: Processes warped previous HQ frame for temporal guidance
- U-Net: Denoises latents with ControlNet residual injection
- Temporal VAE: Decodes with TPM feature fusion for temporal consistency
- RAFT: Estimates optical flow for frame alignment
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, List, Union
from dataclasses import dataclass, field

from .state import StreamDiffVSRState
from .utils.image_utils import bhwc_to_bchw, bchw_to_bhwc, normalize_to_neg1_1, denormalize_from_neg1_1, pad_to_multiple, crop_to_size
from .utils.flow_utils import flow_warp


@dataclass
class StreamDiffVSRConfig:
    """Configuration for Stream-DiffVSR pipeline."""

    scale_factor: int = 4
    latent_channels: int = 4
    latent_scale: int = 4  # Spatial downscale in latent space (TemporalAutoencoderTiny uses 4x)
    num_inference_steps: int = 4
    vae_scaling_factor: float = 1.0  # AutoEncoderTiny uses 1.0
    guidance_scale: float = 0.0  # No CFG by default (as per upstream)
    controlnet_conditioning_scale: float = 1.0


class StreamDiffVSRPipeline:
    """
    Stream-DiffVSR inference pipeline.

    Implements the auto-regressive diffusion framework for video
    super-resolution with temporal guidance from previous frames.

    CRITICAL: This pipeline owns the denoising loop. We cannot use
    ComfyUI's KSampler because ControlNet injects features mid-network.
    
    Key architecture:
    - Flow computed on bicubic 4x upscaled images (not LQ resolution)
    - ControlNet provides temporal guidance via warped previous HQ frame
    - Temporal VAE decodes with TPM features from warped previous HQ
    """

    def __init__(
        self,
        unet,
        controlnet,
        vae,
        scheduler,
        flow_estimator,
        config: StreamDiffVSRConfig = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        text_encoder=None,
        tokenizer=None,
    ):
        """
        Initialize pipeline.

        Args:
            unet: UNet2DConditionModel for denoising
            controlnet: ControlNet for temporal guidance
            vae: TemporalAutoencoderTiny with TPM
            scheduler: DDIMScheduler
            flow_estimator: RAFT optical flow estimator
            config: Pipeline configuration
            device: Target device
            dtype: Model dtype
            text_encoder: Optional text encoder (for prompts)
            tokenizer: Optional tokenizer (for prompts)
        """
        self.unet = unet
        self.controlnet = controlnet
        self.vae = vae
        self.scheduler = scheduler
        self.flow_estimator = flow_estimator
        self.config = config or StreamDiffVSRConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        # Default empty prompt embeddings (no text conditioning)
        self._empty_prompt_embeds = None

    def to(self, device=None, dtype=None):
        """Move pipeline to device/dtype."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
            
        self.unet = self.unet.to(self.device, self.dtype)
        self.controlnet = self.controlnet.to(self.device, self.dtype)
        self.vae = self.vae.to(self.device, self.dtype)
        if self.flow_estimator is not None:
            self.flow_estimator = self.flow_estimator.to(self.device)
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self.device, self.dtype)
        return self

    def _get_prompt_embeds(self, batch_size: int = 1) -> torch.Tensor:
        """Get empty prompt embeddings for unconditional generation."""
        if self._empty_prompt_embeds is not None:
            embeds = self._empty_prompt_embeds
            if embeds.shape[0] != batch_size:
                embeds = embeds.repeat(batch_size, 1, 1)
            return embeds.to(self.device, self.dtype)
        
        # Create empty prompt embeddings
        # SD x4 Upscaler expects encoder_hidden_states
        if self.text_encoder is not None and self.tokenizer is not None:
            # Use actual text encoder
            text_inputs = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                text_embeds = self.text_encoder(
                    text_inputs.input_ids.to(self.device)
                )[0]
            self._empty_prompt_embeds = text_embeds
        else:
            # Create dummy embeddings (77 tokens, 1024 dim for SDXL-like)
            # Actual dimension depends on the model
            hidden_size = getattr(self.unet.config, 'cross_attention_dim', 1024)
            self._empty_prompt_embeds = torch.zeros(
                1, 77, hidden_size, device=self.device, dtype=self.dtype
            )
        
        return self._empty_prompt_embeds.repeat(batch_size, 1, 1).to(self.device, self.dtype)

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare initial noise latents."""
        latent_h = height // self.config.latent_scale
        latent_w = width // self.config.latent_scale
        
        shape = (batch_size, self.config.latent_channels, latent_h, latent_w)
        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Scale by scheduler's initial sigma
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.inference_mode()
    def compute_flows(
        self,
        upscaled_images: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Compute optical flows between consecutive upscaled frames.

        Args:
            upscaled_images: List of 4x bicubic upscaled images (B, C, H, W)

        Returns:
            List of flow tensors (B, H, W, 2), one per consecutive pair
        """
        print('[Stream-DiffVSR] Computing optical flows...')
        flows = []
        for i in range(1, len(upscaled_images)):
            prev_image = upscaled_images[i - 1]
            cur_image = upscaled_images[i]
            # Flow from prev to current
            flow = self.flow_estimator(cur_image, prev_image)
            flows.append(flow)
        return flows

    @torch.inference_mode()
    def process_frame(
        self,
        lq_frame: torch.Tensor,
        state: StreamDiffVSRState,
        lq_upscaled: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        num_inference_steps: int = 4,
        seed: int = 0,
        guidance_scale: float = 0.0,
        controlnet_conditioning_scale: float = 1.0,
        force_flow_on_lq: bool = False,
    ) -> Tuple[torch.Tensor, StreamDiffVSRState]:
        """
        Process a single frame with ControlNet temporal guidance.

        Args:
            lq_frame: Low-quality input frame (1, H, W, C) in [0, 1] BHWC
            state: Previous frame state
            lq_upscaled: Pre-computed bicubic 4x upscaled LQ (optional)
            flow: Pre-computed optical flow (optional)
            num_inference_steps: Number of denoising steps
            seed: Random seed
            guidance_scale: CFG scale (0 = no CFG, as per upstream)
            controlnet_conditioning_scale: ControlNet strength
            force_flow_on_lq: If True, compute flow on LQ frames (faster). If False, compute on HQ (better quality).

        Returns:
            hq_frame: High-quality output frame (1, H*4, W*4, C) BHWC
            new_state: Updated state for next frame
        """
        # Setup
        generator = torch.Generator(device=self.device).manual_seed(seed)
        B, H, W, C = lq_frame.shape
        scale = self.config.scale_factor
        target_h, target_w = H * scale, W * scale
        
        # Warn if sizes aren't multiples of latent_scale (8)
        # This can cause output size mismatches with the bicubic guide
        latent_scale = self.config.latent_scale
        if target_h % latent_scale != 0 or target_w % latent_scale != 0:
            import warnings
            warnings.warn(
                f"Input size {H}x{W} (target {target_h}x{target_w}) is not a multiple of {latent_scale}. "
                f"This may cause slight output size mismatches. Consider resizing input to multiples of {latent_scale // scale}.",
                UserWarning
            )

        # Convert to model format (BCHW, [-1, 1])
        lq_bchw = bhwc_to_bchw(lq_frame).to(self.device, self.dtype)
        lq_normalized = normalize_to_neg1_1(lq_bchw)

        # Bicubic upscale LQ to target resolution (for flow and U-Net input)
        if lq_upscaled is None:
            lq_upscaled = F.interpolate(
                lq_bchw, scale_factor=scale, mode='bicubic', align_corners=False
            )
        else:
            lq_upscaled = lq_upscaled.to(self.device, self.dtype)
            # Ensure lq_upscaled matches expected target size
            _, _, up_h, up_w = lq_upscaled.shape
            if up_h != target_h or up_w != target_w:
                lq_upscaled = F.interpolate(
                    lq_upscaled, size=(target_h, target_w), mode='bicubic', align_corners=False
                )
        
        lq_upscaled_normalized = normalize_to_neg1_1(lq_upscaled)

        # Handle temporal guidance
        warped_prev_hq = None
        dec_temporal_features = None

        if state.has_previous:
            prev_hq = state.previous_hq.to(self.device, self.dtype)

            # Compute flow if not provided
            if flow is None:
                if force_flow_on_lq:
                    # Compute flow on LQ frames (fast)
                    if state.previous_lq_upscaled is not None:
                         # Use raw LQ (ensure range [0, 255] for RAFT)
                         # lq_bchw is normalized [-1, 1]. RAFT FlowEstimator handles normalization if range is [-1, 1].
                         
                         # Previous LQ needs to be derived from upscaled version (since we don't store raw LQ in state)
                         # Downsampling is fast.
                         prev_lq = F.interpolate(state.previous_lq_upscaled.to(self.device, self.dtype), 
                                                scale_factor=1/self.config.scale_factor, mode='bilinear')
                         
                         # Current LQ
                         curr_lq = lq_bchw
                         
                         flow_lq = self.flow_estimator(curr_lq, prev_lq)
                         
                         # Upscale flow to HQ resolution
                         from .utils.flow_utils import upscale_flow
                         flow = upscale_flow(flow_lq, scale_factor=self.config.scale_factor)
                    else:
                        flow = torch.zeros(B, 2, target_h, target_w, device=self.device, dtype=self.dtype)
                else:
                    # Compute flow on Upscaled LQ frames (slow, standard)
                    prev_lq_upscaled = state.previous_lq_upscaled.to(self.device, self.dtype)
                    flow = self.flow_estimator(lq_upscaled, prev_lq_upscaled)

            # Warp previous HQ to current frame
            warped_prev_hq = flow_warp(prev_hq, flow, interp_mode='bilinear')

            # Extract temporal features for VAE decoder TPM
            enc_layer_features = self.vae.encode(warped_prev_hq, return_features_only=True)
            if isinstance(enc_layer_features, list):
                dec_temporal_features = enc_layer_features[::-1]  # Reverse for decoder order
            else:
                dec_temporal_features = None

        # Prepare initial latents
        latents = self._prepare_latents(B, target_h, target_w, generator)

        # Get prompt embeddings (empty for unconditional)
        prompt_embeds = self._get_prompt_embeds(B)
        do_cfg = guidance_scale > 1.0

        if do_cfg:
            # Duplicate embeddings for CFG
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])

        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Reset VAE temporal state
        if hasattr(self.vae, 'reset_temporal_condition'):
            self.vae.reset_temporal_condition()

        # ===== DENOISING LOOP =====
        for i, t in enumerate(timesteps):
            # Prepare latent input
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Concatenate LQ image to latents (SD x4 Upscaler style)
            # The upscaled LQ provides structural guidance
            lq_for_concat = lq_upscaled_normalized
            if do_cfg:
                lq_for_concat = torch.cat([lq_for_concat] * 2)
            
            # Check if U-Net expects concatenated input
            unet_in_channels = getattr(self.unet.config, 'in_channels', 4)
            if unet_in_channels > self.config.latent_channels:
                # Encode LQ to latent space for concatenation
                lq_latent = self.vae.encode(lq_upscaled_normalized)
                if do_cfg:
                    lq_latent = torch.cat([lq_latent] * 2)
                latent_model_input = torch.cat([latent_model_input, lq_latent], dim=1)

            # ControlNet (temporal guidance from warped previous HQ)
            if warped_prev_hq is not None and not state.frame_index == 0:
                controlnet_cond = warped_prev_hq
                if do_cfg:
                    controlnet_cond = torch.cat([controlnet_cond] * 2)
                
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
            else:
                # First frame: no temporal conditioning
                down_block_res_samples = None
                mid_block_res_sample = None

            # U-Net prediction with ControlNet injection
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            # CFG guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            step_output = self.scheduler.step(noise_pred, t, latents)
            latents = step_output.prev_sample

        # ===== DECODE with Temporal Features =====
        # Decode latents with TPM temporal feature fusion
        scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.0)
        latents_scaled = latents / scaling_factor
        
        hq_bchw = self.vae.decode(latents_scaled, temporal_features=dec_temporal_features)
        
        # Reset temporal state after decode
        if hasattr(self.vae, 'reset_temporal_condition'):
            self.vae.reset_temporal_condition()

        # Clamp and convert to [0, 1]
        hq_bchw = torch.clamp(hq_bchw, -1, 1)
        hq_bchw = denormalize_from_neg1_1(hq_bchw)

        # Convert back to ComfyUI format (BHWC, CPU, float32)
        hq_frame = bchw_to_bhwc(hq_bchw).cpu().float()

        # Create new state
        # Store HQ in normalized [-1, 1] range for next frame's warping
        new_state = StreamDiffVSRState(
            previous_hq=normalize_to_neg1_1(hq_bchw).cpu().float(),  # BCHW, [-1,1]
            previous_lq_upscaled=lq_upscaled.cpu().float(),  # BCHW, [0,1]
            frame_index=state.frame_index + 1,
            metadata=state.metadata.copy() if state.metadata else {},  # Preserve metadata
        )

        return hq_frame, new_state

    @torch.inference_mode()
    def __call__(
        self,
        images: torch.Tensor,
        state: Optional[StreamDiffVSRState] = None,
        num_inference_steps: int = 4,
        seed: int = 0,
        guidance_scale: float = 0.0,
        controlnet_conditioning_scale: float = 1.0,
        force_flow_on_lq: bool = False,
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
            guidance_scale: CFG scale (0 = disabled, as per upstream)
            controlnet_conditioning_scale: ControlNet strength
            progress_callback: Optional callback(frame_idx, total_frames)

        Returns:
            hq_images: Upscaled frames (B, H*4, W*4, C) BHWC
            final_state: State after last frame
        """
        num_frames = images.shape[0]
        scale = self.config.scale_factor

        # Initialize state if not provided
        if state is None:
            state = StreamDiffVSRState()

        # Pre-compute all LQ upscaled images and flows
        print('[Stream-DiffVSR] Preparing frames...')
        lq_list = []  # Original LQ frames (for fast flow)
        lq_upscaled_list = []  # Upscaled LQ frames
        for i in range(num_frames):
            frame = images[i:i+1]
            lq_bchw = bhwc_to_bchw(frame).to(self.device, self.dtype)
            lq_list.append(lq_bchw)
            lq_upscaled = F.interpolate(
                lq_bchw, scale_factor=scale, mode='bicubic', align_corners=False
            )
            lq_upscaled_list.append(lq_upscaled)

        # Compute flows
        print('[Stream-DiffVSR] Computing optical flows...')
        if force_flow_on_lq:
            # FAST: Compute flow on original LQ resolution, then upscale
            if state.has_previous:
                # Downsample previous upscaled LQ to original resolution
                prev_lq = F.interpolate(
                    state.previous_lq_upscaled.to(self.device, self.dtype),
                    scale_factor=1/scale, mode='bilinear'
                )
                all_lq = [prev_lq] + lq_list
            else:
                all_lq = lq_list
            
            flows_lq = self.compute_flows(all_lq)
            
            # Upscale flows to HQ resolution
            from .utils.flow_utils import upscale_flow
            flows = [upscale_flow(f, scale_factor=scale) for f in flows_lq]
        else:
            # STANDARD: Compute flow on upscaled images (uses more VRAM)
            if state.has_previous:
                all_upscaled = [state.previous_lq_upscaled.to(self.device, self.dtype)] + lq_upscaled_list
            else:
                all_upscaled = lq_upscaled_list
            
            flows = self.compute_flows(all_upscaled)

        # Process frames
        hq_frames = []
        for i in range(num_frames):
            frame = images[i:i+1]
            lq_upscaled = lq_upscaled_list[i]
            
            # Get flow for this frame
            if state.has_previous or i > 0:
                flow_idx = i if state.has_previous else i - 1
                if flow_idx >= 0 and flow_idx < len(flows):
                    flow = flows[flow_idx]
                else:
                    flow = None
            else:
                flow = None

            # Process with temporal guidance
            # Note: We don't pass lq_upscaled - process_frame computes it to ensure size consistency
            hq_frame, state = self.process_frame(
                frame,
                state,
                lq_upscaled=None,  # Let process_frame compute to ensure sizes match
                flow=flow,
                num_inference_steps=num_inference_steps,
                seed=seed + i,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                force_flow_on_lq=force_flow_on_lq,
            )

            hq_frames.append(hq_frame)

            if progress_callback is not None:
                progress_callback(i + 1, num_frames)

        # Stack frames back to batch
        hq_images = torch.cat(hq_frames, dim=0)

        return hq_images, state
