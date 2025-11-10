# Prediction interface for Cog ⚙️
# https://cog.run/python

import subprocess
import tempfile
import os
import json
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
            choices=["create_preview_video", "boomerang", "reencode_for_web"]
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
        elif task == "reencode_for_web":
            return self._reencode_for_web(video, start_time, end_time)
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
        1. Reducing resolution to 480p
        2. Using a moderate bitrate while maintaining good quality
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
    
    def _get_video_fps(self, input_path: str) -> float:
        """Get the frame rate of the input video"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                return float(num) / float(den)
            return float(fps_str)
        except (subprocess.CalledProcessError, ValueError, ZeroDivisionError):
            return 30.0  # Default to 30 fps if we can't determine
    
    def _get_video_resolution(self, input_path: str) -> tuple[int, int]:
        """Get the resolution (width, height) of the input video"""
        try:
            # Use JSON output for more reliable parsing
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                input_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            if "streams" in data and len(data["streams"]) > 0:
                stream = data["streams"][0]
                width = stream.get("width")
                height = stream.get("height")
                if width and height:
                    return (int(width), int(height))
            return (1920, 1080)  # Default to 1080p if we can't determine
        except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError):
            return (1920, 1080)  # Default to 1080p if we can't determine
    
    def _get_video_bitrate(self, input_path: str) -> int:
        """Get the bitrate of the input video in bps"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            bitrate_str = result.stdout.strip()
            if bitrate_str and bitrate_str != "N/A":
                return int(bitrate_str)
            return None
        except (subprocess.CalledProcessError, ValueError):
            return None
    
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
                "-vf", "scale_cuda=-2:480",  # GPU-accelerated scaling to 480p
            ])
        else:
            cmd.extend([
                "-vf", "scale=-2:480",  # CPU scaling for non-h264 inputs
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
            "-b:v", "700k",  # Target video bitrate
            "-maxrate", "1000k",  # Maximum video bitrate
            "-bufsize", "2000k",  # Buffer size
            "-g", "30",  # Keyframe interval
            "-bf", "0",  # No B-frames for baseline profile compatibility
            "-movflags", "+faststart",  # Enable fast start for web playback
            "-c:a", "aac",  # Use AAC audio codec
            "-b:a", "128k",  # Audio bitrate
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
            "-vf", "scale=-2:480",  # Scale to 480p maintaining aspect ratio
            "-c:v", "libx264",  # Use H.264 codec
            "-preset", "medium",  # Balance between encoding speed and compression
            "-crf", "26",  # Constant Rate Factor (lower = better quality)
            "-profile:v", "baseline",  # Most compatible H.264 profile
            "-movflags", "+faststart",  # Enable fast start for web playback
            "-g", "30",  # Add keyframe every 30 frames
            "-sc_threshold", "0",  # Disable scene change detection
            "-keyint_min", "30",  # Minimum keyframe interval
            "-b:v", "700k",  # Target video bitrate
            "-maxrate", "1000k",  # Maximum video bitrate
            "-bufsize", "2000k",  # Buffer size
            "-c:a", "aac",  # Use AAC audio codec
            "-b:a", "128k",  # Audio bitrate
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
    
    def _reencode_for_web(
        self,
        video: Path,
        start_time: float,
        end_time: float
    ) -> Path:
        """Re-encode video optimized for web streaming
        
        This creates a video optimized for web by:
        1. Using H.264 codec with medium preset for quality/speed balance
        2. Adding frequent keyframes for fast seeking (every 2 seconds)
        3. Using faststart for progressive download
        4. Using CRF 23 for good quality while keeping file sizes smaller
        5. Using high profile for better compression efficiency
        6. Capping resolution to 2K (1440p) max for web efficiency
        7. Capping bitrate to ensure smooth streaming on most connections
        8. Maintaining good visual quality while reducing file size
        """
        # Create temporary output file
        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_file.name
        output_file.close()
        
        # Check input video properties
        input_codec = self._get_video_codec(str(video))
        fps = self._get_video_fps(str(video))
        width, height = self._get_video_resolution(str(video))
        input_bitrate = self._get_video_bitrate(str(video))
        
        print(f"Input video: codec={input_codec}, FPS={fps}, resolution={width}x{height}, bitrate={input_bitrate}")
        
        # Calculate keyframe interval for 2 seconds
        keyframe_interval = int(fps * 2)
        
        # Determine target resolution - cap at 2K (1440p) for web efficiency
        # If input is larger, scale down while maintaining aspect ratio
        max_height = 1440
        if height > max_height:
            # Calculate new width maintaining aspect ratio
            new_height = max_height
            new_width = int(width * (max_height / height))
            # Ensure width is even (required by H.264)
            new_width = new_width if new_width % 2 == 0 else new_width - 1
            scale_filter = f"scale={new_width}:{new_height}"
            print(f"Scaling down from {width}x{height} to {new_width}x{new_height} for web optimization")
        else:
            scale_filter = None
            new_width, new_height = width, height
        
        # Determine target bitrate based on resolution
        # Use conservative bitrates to reduce file size
        if new_height >= 1440:
            target_bitrate = "8M"
            max_bitrate = "12M"
        elif new_height >= 1080:
            target_bitrate = "5M"
            max_bitrate = "8M"
        elif new_height >= 720:
            target_bitrate = "3M"
            max_bitrate = "5M"
        else:
            target_bitrate = "2M"
            max_bitrate = "3M"
        
        # If input bitrate is lower, try to match or be slightly more efficient
        if input_bitrate:
            input_bitrate_mbps = input_bitrate / 1000000
            # Use 80% of input bitrate as target, but not less than our minimums
            calculated_target = max(2.0, input_bitrate_mbps * 0.8)
            if calculated_target < float(target_bitrate.replace('M', '')):
                target_bitrate = f"{int(calculated_target)}M"
                max_bitrate = f"{int(calculated_target * 1.5)}M"
                print(f"Adjusting bitrate based on input: target={target_bitrate}, max={max_bitrate}")
        
        # Try hardware-accelerated encoding first
        if self.gpu_available:
            success = self._encode_web_with_hardware(
                str(video), output_path, start_time, end_time, input_codec, 
                keyframe_interval, scale_filter, target_bitrate, max_bitrate
            )
            if success:
                print("Successfully encoded web-optimized video with hardware acceleration")
                return Path(output_path)
        
        # Fallback to software encoding
        print("Using software encoding for web optimization")
        self._encode_web_with_software(
            str(video), output_path, start_time, end_time, keyframe_interval,
            scale_filter, target_bitrate, max_bitrate
        )
        
        return Path(output_path)
    
    def _encode_web_with_hardware(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        input_codec: str,
        keyframe_interval: int,
        scale_filter: str,
        target_bitrate: str,
        max_bitrate: str
    ) -> bool:
        """Try to encode web-optimized video using hardware acceleration"""
        cmd = ["ffmpeg", "-y"]
        
        # Use hardware decoder if input is h264
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
        
        # Add input
        cmd.extend(["-i", input_path])
        
        # Apply scaling if needed (using GPU-accelerated scaling)
        if scale_filter and input_codec == "h264":
            # Use GPU scaling for h264 input
            scale_filter_gpu = scale_filter.replace("scale=", "scale_cuda=")
            cmd.extend(["-vf", scale_filter_gpu])
        elif scale_filter:
            # Use CPU scaling for non-h264 input
            cmd.extend(["-vf", scale_filter])
        
        # Calculate bufsize (typically 2x maxrate)
        maxrate_value = int(max_bitrate.replace('M', ''))
        bufsize = f"{maxrate_value * 2}M"
        
        # Hardware encoding parameters optimized for web
        # Use CQ 23 for good quality with smaller file sizes
        cmd.extend([
            "-c:v", "h264_nvenc",  # NVIDIA hardware encoder
            "-preset", "p4",  # Medium preset (p4) for quality/speed balance
            "-profile:v", "high",  # High profile for better compression efficiency
            "-rc", "vbr",  # Variable bitrate mode
            "-cq", "23",  # Constant quality (similar to CRF 23) for good quality with smaller files
            "-rc-lookahead", "32",  # Lookahead for better quality
            "-spatial_aq", "1",  # Spatial adaptive quantization
            "-temporal_aq", "1",  # Temporal adaptive quantization
            "-b:v", target_bitrate,  # Target bitrate (optimized for web)
            "-maxrate", max_bitrate,  # Maximum bitrate cap for smooth streaming
            "-bufsize", bufsize,  # Buffer size
            "-g", str(keyframe_interval),  # Keyframe interval (every 2 seconds)
            "-force_key_frames", "expr:gte(t,n_forced*2)",  # Force keyframes every 2 seconds (more precise)
            "-bf", "3",  # B-frames for high profile
            "-movflags", "+faststart",  # Enable faststart for progressive download
            "-c:a", "aac",  # Use AAC audio codec
            "-b:a", "128k",  # Audio bitrate (reduced for smaller file size)
            "-ac", "2",  # 2 audio channels (stereo)
            "-ar", "48000",  # Audio sample rate
            output_path
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Hardware web encoding failed: {e.stderr.decode('utf-8', errors='replace')}")
            return False
    
    def _encode_web_with_software(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        keyframe_interval: int,
        scale_filter: str,
        target_bitrate: str,
        max_bitrate: str
    ):
        """Encode web-optimized video using software encoder"""
        cmd = ["ffmpeg", "-y"]
        
        # Add trimming parameters
        if start_time != 0:
            cmd.extend(["-ss", str(start_time)])
        
        if end_time != -1:
            cmd.extend(["-to", str(end_time)])
        
        # Add input
        cmd.extend(["-i", input_path])
        
        # Build video filter chain
        vf_filters = []
        if scale_filter:
            vf_filters.append(scale_filter)
        
        # Add input and encoding parameters optimized for web
        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])
        
        # Calculate bufsize (typically 2x maxrate)
        maxrate_value = int(max_bitrate.replace('M', ''))
        bufsize = f"{maxrate_value * 2}M"
        
        # Calculate maxrate and bufsize values for x264-params
        maxrate_bps = int(max_bitrate.replace('M', '')) * 1000000
        bufsize_bps = int(bufsize.replace('M', '')) * 1000000
        
        cmd.extend([
            "-c:v", "libx264",  # Use H.264 codec
            "-preset", "medium",  # Medium preset for quality/speed balance
            "-crf", "23",  # CRF 23 for good quality while keeping file sizes smaller
            "-profile:v", "high",  # High profile for better compression efficiency
            "-level", "4.0",  # H.264 level 4.0 for compatibility
            "-x264-params", f"vbv-maxrate={maxrate_bps}:vbv-bufsize={bufsize_bps}",  # Enforce strict VBV limits for file size control
            "-movflags", "+faststart",  # Enable faststart for progressive download
            "-g", str(keyframe_interval),  # Keyframe interval (every 2 seconds, in frames)
            "-force_key_frames", "expr:gte(t,n_forced*2)",  # Force keyframes every 2 seconds (more precise)
            "-keyint_min", str(keyframe_interval),  # Minimum keyframe interval
            "-sc_threshold", "0",  # Disable scene change detection (use fixed keyframes)
            "-maxrate", max_bitrate,  # Maximum video bitrate cap for smooth streaming
            "-bufsize", bufsize,  # Buffer size
            "-c:a", "aac",  # Use AAC audio codec
            "-b:a", "128k",  # Audio bitrate (reduced for smaller file size)
            "-ac", "2",  # 2 audio channels (stereo)
            "-ar", "48000",  # Audio sample rate
            output_path
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            error_msg = f"Software web encoding failed: {e.stderr.decode('utf-8', errors='replace')}"
            print(error_msg)
            raise RuntimeError(error_msg)
