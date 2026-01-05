"""
All-in-one video upscale node for Stream-DiffVSR.

This node handles entire video files with automatic chunking,
making it the recommended approach for processing long videos.
"""

import os
import subprocess
import tempfile
import torch
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

import folder_paths
from comfy.utils import ProgressBar

from ..stream_diffvsr.state import StreamDiffVSRState
from ..stream_diffvsr.pipeline import StreamDiffVSRPipeline


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(",")
        
        if len(parts) >= 4:
            width, height = int(parts[0]), int(parts[1])
            fps_parts = parts[2].split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            frame_count = int(parts[3])
        else:
            # Fallback: count frames manually
            width, height = int(parts[0]), int(parts[1])
            fps_parts = parts[2].split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            frame_count = count_frames_fallback(video_path)
            
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get video info: {e}")


def count_frames_fallback(video_path: str) -> int:
    """Count frames using ffprobe if nb_read_packets fails."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",  
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def load_video_frames(
    video_path: str,
    start_frame: int,
    num_frames: int,
) -> torch.Tensor:
    """
    Load a batch of frames from video using ffmpeg.
    
    Returns tensor in ComfyUI format: (B, H, W, C) float32 [0, 1]
    """
    # Get video dimensions
    info = get_video_info(video_path)
    width, height = info["width"], info["height"]
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='between(n,{start_frame},{start_frame + num_frames - 1})'",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ]
    
    result = subprocess.run(cmd, capture_output=True, check=True)
    
    # Parse raw video data
    frame_size = width * height * 3
    raw_data = result.stdout
    actual_frames = len(raw_data) // frame_size
    
    if actual_frames == 0:
        raise RuntimeError(f"No frames loaded from {video_path} at position {start_frame}")
    
    # Convert to numpy then torch
    frames = np.frombuffer(raw_data[:actual_frames * frame_size], dtype=np.uint8)
    frames = frames.reshape(actual_frames, height, width, 3)
    frames = torch.from_numpy(frames.copy()).float() / 255.0
    
    return frames


def save_frames_to_video(
    frames: torch.Tensor,
    output_path: str,
    fps: float,
    append: bool = False,
) -> None:
    """
    Save frames to video file using ffmpeg.
    
    Args:
        frames: (B, H, W, C) float32 [0, 1] tensor
        output_path: Output video path
        fps: Frame rate
        append: If True, append to existing file
    """
    # Convert to numpy uint8
    frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
    b, h, w, c = frames_np.shape
    
    if append and os.path.exists(output_path):
        # Append mode: use temporary file and concat
        temp_path = output_path + ".temp.mp4"
        
        # Write new frames to temp file
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo", 
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            temp_path
        ]
        
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = proc.communicate(input=frames_np.tobytes())
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed: {stderr.decode('utf-8', errors='replace')}")
        
        # Concat original with new frames
        concat_list = output_path + ".concat.txt"
        with open(concat_list, "w") as f:
            f.write(f"file '{os.path.abspath(output_path)}'\n")
            f.write(f"file '{os.path.abspath(temp_path)}'\n")
        
        final_temp = output_path + ".final.mp4"
        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            final_temp
        ]
        subprocess.run(concat_cmd, check=True, capture_output=True)
        
        # Replace original with concatenated version
        os.remove(output_path)
        os.rename(final_temp, output_path)
        os.remove(temp_path)
        os.remove(concat_list)
    else:
        # First batch: create new file
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = proc.communicate(input=frames_np.tobytes())
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed: {stderr.decode('utf-8', errors='replace')}")


class StreamDiffVSR_UpscaleVideo:
    """
    All-in-one video upscaling node.
    
    Supports two modes:
    1. File Mode (Recommended for long videos):
       - Input: video_path
       - Output: output_path (streams to disk)
       - Memory: Very low VRAM/RAM usage
       
    2. Tensor Mode (Standard ComfyUI workflow):
       - Input: images (from Load Video/Load Image)
       - Output: images (to Save Video/Preview)
       - Memory: High RAM usage (stores all frames), use for short clips
       
    Both modes use automatic chunking to manage VRAM during inference.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": (
                    "STREAM_DIFFVSR_PIPE",
                    {
                        "tooltip": "Pipeline from StreamDiffVSR_Loader",
                    },
                ),
            },
            "optional": {
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Input video path (File Mode)",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Input frames tensor (Tensor Mode)",
                    },
                ),
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Output path (File Mode only)",
                    },
                ),
                "frames_per_batch": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Frames to process at once (VRAM optimization)",
                    },
                ),
                "start_frame": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "First frame to process (0-indexed)",
                    },
                ),
                "end_frame": (
                    "INT",
                    {
                        "default": -1,
                        "tooltip": "Last frame to process (-1 = all)",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Denoising steps",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Random seed",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "CFG scale (0 = disabled)",
                    },
                ),
                "controlnet_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "ControlNet strength",
                    },
                ),
                "force_flow_on_lq": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Compute optical flow on low-res frames. "
                            "Much faster and saves VRAM, with minimal quality loss."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "output_path")
    OUTPUT_NODE = True
    FUNCTION = "upscale_video"
    CATEGORY = "StreamDiffVSR"
    DESCRIPTION = "Upscale video (File or Tensor) with auto-chunking 4x"

    def upscale_video(
        self,
        pipe: StreamDiffVSRPipeline,
        video_path: str = "",
        images: Optional[torch.Tensor] = None,
        output_path: str = "",
        frames_per_batch: int = 16,
        start_frame: int = 0,
        end_frame: int = -1,
        num_inference_steps: int = 4,
        seed: int = 0,
        guidance_scale: float = 0.0,
        controlnet_scale: float = 1.0,
        force_flow_on_lq: bool = False,
    ) -> Tuple[Optional[torch.Tensor], str]:
        
        # Mode detection
        has_images = images is not None
        has_video = video_path is not None and video_path.strip() != "" and os.path.exists(video_path)
        
        if has_images:
            # --- Tensor Mode ---
            return self._upscale_tensor(
                pipe, images, frames_per_batch, num_inference_steps,
                seed, guidance_scale, controlnet_scale, force_flow_on_lq
            )
        elif has_video:
            # --- File Mode ---
            return self._upscale_file(
                pipe, video_path, output_path, frames_per_batch,
                start_frame, end_frame, num_inference_steps,
                seed, guidance_scale, controlnet_scale, force_flow_on_lq
            )
        else:
            raise ValueError("No input provided. Please provide either 'images' (Tensor) or 'video_path' (File).")

    def _upscale_tensor(
        self, pipe, images, frames_per_batch, num_inference_steps,
        seed, guidance_scale, controlnet_scale, force_flow_on_lq
    ):
        total_frames = images.shape[0]
        print(f"[Stream-DiffVSR] Processing {total_frames} frames (Tensor Mode)")
        
        results = []
        state = StreamDiffVSRState()
        num_batches = (total_frames + frames_per_batch - 1) // frames_per_batch
        pbar = ProgressBar(total_frames)
        
        for batch_idx in range(num_batches):
            start = batch_idx * frames_per_batch
            end = min(start + frames_per_batch, total_frames)
            batch_frames = images[start:end]
            
            hq_batch, state = pipe(
                batch_frames,
                state=state,
                num_inference_steps=num_inference_steps,
                seed=seed + start,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                force_flow_on_lq=force_flow_on_lq,
            )
            
            # Store inputs on CPU to save VRAM
            results.append(hq_batch.cpu())
            pbar.update(end - start)
            
            del batch_frames, hq_batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        final_output = torch.cat(results, dim=0)
        return (final_output, "")

    def _upscale_file(
        self, pipe, video_path, output_path, frames_per_batch,
        start_frame, end_frame, num_inference_steps,
        seed, guidance_scale, controlnet_scale, force_flow_on_lq
    ):
        # ... Reuse existing logic ...
        # Since I'm replacing the whole class, I need to copy the logic here.
        
        # Get video info
        video_info = get_video_info(video_path)
        total_frames = video_info["frame_count"]
        fps = video_info["fps"]
        
        if end_frame < 0:
            end_frame = total_frames - 1
        end_frame = min(end_frame, total_frames - 1)
        frames_to_process = end_frame - start_frame + 1
        
        # Generate output path if not provided
        if not output_path:
            input_name = Path(video_path).stem
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, f"{input_name}_4x.mp4")
            
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
            
        print(f"[Stream-DiffVSR] Streaming {frames_to_process} frames to {output_path}")
        
        num_batches = (frames_to_process + frames_per_batch - 1) // frames_per_batch
        pbar = ProgressBar(frames_to_process)
        state = StreamDiffVSRState()
        
        for batch_idx in range(num_batches):
            batch_start = start_frame + (batch_idx * frames_per_batch)
            batch_frames = min(frames_per_batch, end_frame - batch_start + 1)
            
            # Load frames
            frames = load_video_frames(video_path, batch_start, batch_frames)
            
            # Process
            hq_frames, state = pipe(
                frames,
                state=state,
                num_inference_steps=num_inference_steps,
                seed=seed + batch_start,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                force_flow_on_lq=force_flow_on_lq,
            )
            
            # Save
            save_frames_to_video(
                hq_frames,
                output_path,
                fps,
                append=(batch_idx > 0)
            )
            
            pbar.update(batch_frames)
            
            del frames, hq_frames
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return (None, output_path)
