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
        proc.stdin.write(frames_np.tobytes())
        proc.stdin.close()
        proc.wait()
        
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
        proc.stdin.write(frames_np.tobytes())
        proc.stdin.close()
        proc.wait()


class StreamDiffVSR_UpscaleVideo:
    """
    All-in-one video upscaling node.
    
    Processes an entire video file with automatic chunking to manage
    VRAM usage. This is the recommended node for long video upscaling.
    
    The node:
    1. Reads the input video in batches
    2. Upscales each batch with temporal consistency
    3. Streams results to output file
    4. Maintains state between batches automatically
    
    Note: Audio is not copied. Use VHS or ffmpeg to merge audio separately.
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
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to input video file",
                    },
                ),
            },
            "optional": {
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Output video path. If empty, saves to ComfyUI output folder "
                            "with _4x suffix."
                        ),
                    },
                ),
                "frames_per_batch": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Frames to process per batch. Lower = less VRAM, "
                            "higher = faster (if VRAM allows)."
                        ),
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
                        "tooltip": "Last frame to process (-1 = all frames)",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Denoising steps. Model is optimized for 4 steps.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Random seed for reproducibility",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "CFG scale. 0 = disabled (default).",
                    },
                ),
                "controlnet_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "ControlNet conditioning strength.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    FUNCTION = "upscale_video"
    CATEGORY = "StreamDiffVSR"
    DESCRIPTION = (
        "Upscale entire video 4x with automatic chunking. "
        "Recommended for long videos."
    )

    def upscale_video(
        self,
        pipe: StreamDiffVSRPipeline,
        video_path: str,
        output_path: str = "",
        frames_per_batch: int = 16,
        start_frame: int = 0,
        end_frame: int = -1,
        num_inference_steps: int = 4,
        seed: int = 0,
        guidance_scale: float = 0.0,
        controlnet_scale: float = 1.0,
    ) -> Tuple[str]:
        """
        Upscale an entire video file.
        
        Args:
            pipe: Stream-DiffVSR pipeline
            video_path: Input video file path
            output_path: Output video path (auto-generated if empty)
            frames_per_batch: Frames to process per batch
            start_frame: First frame to process
            end_frame: Last frame to process (-1 = all)
            num_inference_steps: Denoising steps
            seed: Random seed
            guidance_scale: CFG scale
            controlnet_scale: ControlNet strength
            
        Returns:
            Tuple containing output video path
        """
        # Validate input
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Get video info
        print(f"[Stream-DiffVSR] Analyzing video: {video_path}")
        video_info = get_video_info(video_path)
        total_frames = video_info["frame_count"]
        fps = video_info["fps"]
        
        print(f"[Stream-DiffVSR] Video: {video_info['width']}x{video_info['height']}, "
              f"{total_frames} frames @ {fps:.2f} fps")
        
        # Determine frame range
        if end_frame < 0:
            end_frame = total_frames - 1
        end_frame = min(end_frame, total_frames - 1)
        frames_to_process = end_frame - start_frame + 1
        
        # Generate output path if not provided
        if not output_path:
            input_name = Path(video_path).stem
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, f"{input_name}_4x.mp4")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Remove existing output file
        if os.path.exists(output_path):
            os.remove(output_path)
        
        print(f"[Stream-DiffVSR] Processing frames {start_frame}-{end_frame} "
              f"({frames_to_process} frames)")
        print(f"[Stream-DiffVSR] Output: {output_path}")
        print(f"[Stream-DiffVSR] Batch size: {frames_per_batch} frames")
        
        # Calculate number of batches
        num_batches = (frames_to_process + frames_per_batch - 1) // frames_per_batch
        
        # Initialize progress bar
        pbar = ProgressBar(frames_to_process)
        
        # Initialize state for temporal consistency
        state = StreamDiffVSRState()
        processed_frames = 0
        
        for batch_idx in range(num_batches):
            batch_start = start_frame + (batch_idx * frames_per_batch)
            batch_frames = min(frames_per_batch, end_frame - batch_start + 1)
            
            print(f"\n[Stream-DiffVSR] Batch {batch_idx + 1}/{num_batches}: "
                  f"frames {batch_start}-{batch_start + batch_frames - 1}")
            
            # Load frames
            frames = load_video_frames(video_path, batch_start, batch_frames)
            
            # Process with pipeline
            hq_frames, state = pipe(
                frames,
                state=state,
                num_inference_steps=num_inference_steps,
                seed=seed + batch_idx,  # Increment seed per batch
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
            )
            
            # Save frames to output video
            save_frames_to_video(
                hq_frames,
                output_path,
                fps,
                append=(batch_idx > 0)
            )
            
            # Update progress
            processed_frames += batch_frames
            pbar.update(batch_frames)
            
            # Clear some VRAM
            del frames, hq_frames
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n[Stream-DiffVSR] Complete! Saved to: {output_path}")
        
        return (output_path,)
