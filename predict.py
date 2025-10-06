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
        video: Path = Input(description="Input video file URL to process"),
        task: str = Input(
            description="Video processing task to perform",
            default="create_preview_video",
            choices=["create_preview_video", "boomerang"]
        ),
    ) -> Path:
        """Process video with selected task using hardware-accelerated encoding when available"""
        
        # Default parameters
        start_time = 0
        end_time = -1
        preset = "medium"
        bitrate = "20M"
        
        if task == "create_preview_video":
            return self._create_preview_video(video, start_time, end_time, preset, bitrate)
        elif task == "boomerang":
            return self._create_boomerang(video, start_time, end_time, preset, bitrate)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _create_preview_video(
        self,
        video: Path,
        start_time: float,
        end_time: float,
        preset: str,
        bitrate: str
    ) -> Path:
        """Create a low resolution preview video optimized for web seeking
        
        This creates a smaller, more efficient version of the video by:
        1. Reducing resolution to 360p
        2. Using a lower bitrate while maintaining decent quality
        3. Adding more frequent keyframes for better seeking
        4. Using H.264 codec for compatibility
        5. Optimizing audio for web streaming
        """
        # Create temporary output file
        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_file.name
        output_file.close()
        
        # Check input video codec
        input_codec = self._get_video_codec(str(video))
        print(f"Input video codec: {input_codec}")
        
        # Try hardware-accelerated encoding first
        # Note: For small videos or short clips, CPU encoding might be faster due to
        # GPU initialization overhead. The hardware version uses GPU scaling (scale_cuda)
        # to avoid CPU bottlenecks when scaling video.
        if self.gpu_available:
            success = self._encode_preview_with_hardware(
                str(video), output_path, start_time, end_time, input_codec
            )
            if success:
                print("Successfully encoded preview with hardware acceleration")
                return Path(output_path)
        
        # Fallback to software encoding
        print("Using software encoding for preview")
        self._encode_preview_with_software(
            str(video), output_path, start_time, end_time
        )
        
        return Path(output_path)
    
    def _create_boomerang(
        self,
        video: Path,
        start_time: float,
        end_time: float,
        preset: str,
        bitrate: str
    ) -> Path:
        """Create a boomerang effect (forward + reverse playback)"""
        # Create temporary files
        trimmed_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        trimmed_path = trimmed_file.name
        trimmed_file.close()
        
        reversed_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        reversed_path = reversed_file.name
        reversed_file.close()
        
        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_file.name
        output_file.close()
        
        try:
            # First, trim and encode the video
            input_codec = self._get_video_codec(str(video))
            
            # Create trimmed version
            if self.gpu_available:
                success = self._encode_with_hardware(
                    str(video), trimmed_path, start_time, end_time, preset, bitrate, input_codec
                )
                if not success:
                    self._encode_with_software(
                        str(video), trimmed_path, start_time, end_time, preset, bitrate
                    )
            else:
                self._encode_with_software(
                    str(video), trimmed_path, start_time, end_time, preset, bitrate
                )
            
            # Create reversed version
            reverse_cmd = [
                "ffmpeg", "-y",
                "-i", trimmed_path,
                "-vf", "reverse",
                "-af", "areverse"
            ]
            
            if self.gpu_available:
                reverse_cmd.extend([
                    "-c:v", "h264_nvenc",
                    "-preset", preset,
                    "-b:v", bitrate
                ])
            else:
                reverse_cmd.extend([
                    "-c:v", "libx264",
                    "-preset", preset,
                    "-b:v", bitrate
                ])
            
            reverse_cmd.extend([
                "-c:a", "aac",
                "-b:a", "128k",
                reversed_path
            ])
            
            subprocess.run(reverse_cmd, check=True, capture_output=True)
            
            # Create concat file
            concat_file = tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False)
            concat_file.write(f"file '{trimmed_path}'\n")
            concat_file.write(f"file '{reversed_path}'\n")
            concat_file.close()
            
            # Concatenate forward and reverse
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file.name,
                "-c", "copy",
                output_path
            ]
            
            subprocess.run(concat_cmd, check=True, capture_output=True)
            
            # Cleanup temporary files
            os.unlink(trimmed_path)
            os.unlink(reversed_path)
            os.unlink(concat_file.name)
            
            return Path(output_path)
            
        except subprocess.CalledProcessError as e:
            # Cleanup on error
            for path in [trimmed_path, reversed_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
            error_msg = f"Boomerang creation failed: {e.stderr.decode('utf-8', errors='replace')}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
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
    
    def _encode_preview_with_hardware(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        input_codec: str
    ) -> bool:
        """Try to encode preview using hardware acceleration"""
        cmd = ["ffmpeg", "-y"]
        
        # Use hardware decoder and hardware scaling if input is h264
        if input_codec == "h264":
            cmd.extend([
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-c:v", "h264_cuvid"
            ])
        
        # Add trimming parameters
        if start_time != 0:
            cmd.extend(["-ss", str(start_time)])
        
        if end_time != -1:
            cmd.extend(["-to", str(end_time)])
        
        # Add input and use hardware scaling
        cmd.extend(["-i", input_path])
        
        # Use GPU-accelerated scaling if we have hardware decoding
        if input_codec == "h264":
            cmd.extend([
                "-vf", "scale_cuda=-2:360",  # GPU-accelerated scaling to 360p
            ])
        else:
            cmd.extend([
                "-vf", "scale=-2:360",  # CPU scaling for non-h264 inputs
            ])
        
        # Optimized hardware encoding parameters
        cmd.extend([
            "-c:v", "h264_nvenc",  # NVIDIA hardware encoder
            "-preset", "p4",  # P4 preset for balanced quality/speed (p1-p7, p4 is medium)
            "-tune", "hq",  # High quality tuning
            "-rc", "vbr",  # Variable bitrate mode
            "-rc-lookahead", "20",  # Lookahead for better quality
            "-spatial_aq", "1",  # Spatial adaptive quantization
            "-temporal_aq", "1",  # Temporal adaptive quantization
            "-b:v", "300k",  # Target video bitrate (reduced from 500k)
            "-maxrate", "400k",  # Maximum video bitrate (reduced from 600k)
            "-bufsize", "600k",  # Buffer size (reduced from 1000k)
            "-g", "30",  # Keyframe interval
            "-bf", "0",  # No B-frames for baseline profile compatibility
            "-movflags", "+faststart",  # Enable fast start for web playback
            "-c:a", "aac",  # Use AAC audio codec
            "-b:a", "64k",  # Audio bitrate (reduced from 96k)
            "-ac", "2",  # 2 audio channels (stereo)
            "-ar", "44100",  # Audio sample rate
            output_path
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Hardware preview encoding failed: {e.stderr.decode('utf-8', errors='replace')}")
            return False
    
    def _encode_preview_with_software(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ):
        """Encode preview using software encoder with web optimization"""
        cmd = ["ffmpeg", "-y"]
        
        # Add trimming parameters
        if start_time != 0:
            cmd.extend(["-ss", str(start_time)])
        
        if end_time != -1:
            cmd.extend(["-to", str(end_time)])
        
        # Add input and encoding parameters optimized for web preview
        cmd.extend([
            "-i", input_path,
            "-vf", "scale=-2:360",  # Scale to 360p maintaining aspect ratio
            "-c:v", "libx264",  # Use H.264 codec
            "-preset", "medium",  # Balance between encoding speed and compression
            "-crf", "30",  # Constant Rate Factor (increased from 28 for smaller files)
            "-profile:v", "baseline",  # Most compatible H.264 profile
            "-movflags", "+faststart",  # Enable fast start for web playback
            "-g", "30",  # Add keyframe every 30 frames
            "-sc_threshold", "0",  # Disable scene change detection
            "-keyint_min", "30",  # Minimum keyframe interval
            "-b:v", "300k",  # Target video bitrate (reduced from 500k)
            "-maxrate", "400k",  # Maximum video bitrate (reduced from 600k)
            "-bufsize", "600k",  # Buffer size (reduced from 1000k)
            "-c:a", "aac",  # Use AAC audio codec
            "-b:a", "64k",  # Audio bitrate (reduced from 96k)
            "-ac", "2",  # 2 audio channels (stereo)
            "-ar", "44100",  # Audio sample rate
            output_path
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            error_msg = f"Software preview encoding failed: {e.stderr.decode('utf-8', errors='replace')}"
            print(error_msg)
            raise RuntimeError(error_msg)
