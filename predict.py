# Prediction interface for Cog ⚙️
# https://cog.run/python

import subprocess
import tempfile
import os
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Check if NVIDIA GPU is available
        try:
            subprocess.run(["nvidia-smi"], check=True, capture_output=True)
            self.gpu_available = True
            print("NVIDIA GPU detected, hardware acceleration available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.gpu_available = False
            print("No NVIDIA GPU detected, will use software encoding")

    def predict(
        self,
        video: Path = Input(description="Input video file to re-encode"),
        start_time: float = Input(
            description="Start time in seconds for trimming (0 = start of video)",
            default=0,
            ge=0
        ),
        end_time: float = Input(
            description="End time in seconds for trimming (-1 = end of video)",
            default=-1
        ),
        preset: str = Input(
            description="FFmpeg encoding preset (slower = better quality)",
            default="medium",
            choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
        ),
        bitrate: str = Input(
            description="Target bitrate (e.g., '5M' for 5 Mbps, '20M' for 20 Mbps)",
            default="20M"
        ),
    ) -> Path:
        """Re-encode and optionally trim video using hardware-accelerated H264 encoding"""
        
        # Create temporary output file
        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_file.name
        output_file.close()
        
        # Check input video codec
        input_codec = self._get_video_codec(str(video))
        print(f"Input video codec: {input_codec}")
        
        # Try hardware-accelerated encoding first
        if self.gpu_available:
            success = self._encode_with_hardware(
                str(video), output_path, start_time, end_time, preset, bitrate, input_codec
            )
            if success:
                print("Successfully encoded with hardware acceleration")
                return Path(output_path)
        
        # Fallback to software encoding
        print("Using software encoding")
        self._encode_with_software(
            str(video), output_path, start_time, end_time, preset, bitrate
        )
        
        return Path(output_path)
    
    def _get_video_codec(self, input_path: str) -> str:
        """Get the codec of the input video"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def _encode_with_hardware(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        preset: str,
        bitrate: str,
        input_codec: str
    ) -> bool:
        """Try to encode using hardware acceleration"""
        cmd = ["ffmpeg", "-y"]
        
        # Use hardware decoder if input is h264
        if input_codec == "h264":
            cmd.extend(["-c:v", "h264_cuvid"])
        
        # Add trimming parameters
        if start_time != 0:
            cmd.extend(["-ss", str(start_time)])
        
        if end_time != -1:
            cmd.extend(["-to", str(end_time)])
        
        # Add input and encoding parameters
        cmd.extend([
            "-i", input_path,
            "-c:v", "h264_nvenc",  # NVIDIA hardware encoder
            "-preset", preset,
            "-b:v", bitrate,
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Hardware encoding failed: {e.stderr.decode('utf-8', errors='replace')}")
            return False
    
    def _encode_with_software(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        preset: str,
        bitrate: str
    ):
        """Encode using software encoder (fallback)"""
        cmd = ["ffmpeg", "-y"]
        
        # Add trimming parameters
        if start_time != 0:
            cmd.extend(["-ss", str(start_time)])
        
        if end_time != -1:
            cmd.extend(["-to", str(end_time)])
        
        # Add input and encoding parameters
        cmd.extend([
            "-i", input_path,
            "-c:v", "libx264",  # Software encoder
            "-preset", preset,
            "-b:v", bitrate,
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            error_msg = f"Software encoding failed: {e.stderr.decode('utf-8', errors='replace')}"
            print(error_msg)
            raise RuntimeError(error_msg)
