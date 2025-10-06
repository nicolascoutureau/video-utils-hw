import subprocess
import tempfile

import sieve


@sieve.function(
    name="video-reencoder",
    metadata=sieve.Metadata(
        description="Re-encode and trim videos using hardware-accelerated H264 encoding",
    ),
    python_version="3.11",
    python_packages=["opencv-python-headless"],
    system_packages=["ffmpeg"],
    gpu=sieve.gpu.T4(),
)
def reencoder(
    video: sieve.File,
    start_time: float = 0,
    end_time: float = -1,
    preset: str = "medium",
) -> sieve.File:
    """
    Args:
        video: Input video file
        start_time: Start time in seconds for trimming (default: 0 - start of video)
        end_time: End time in seconds for trimming (default: -1 - end of video)
        quality: CRF quality value (lower = better quality, higher = smaller file)
        preset: FFmpeg encoding preset (options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    """
    input_path = video.path
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = output_file.name
    output_file.close()

    # Construct base ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-c:v",
        "h264_cuvid",
    ]

    # Add trimming parameters
    if start_time != 0:
        cmd.extend(["-ss", str(start_time)])

    if end_time != -1:
        cmd.extend(["-to", str(end_time)])

    # Add encoding parameters with hardware acceleration
    cmd.extend(
        [
            "-i",
            input_path,
            "-c:v",
            "h264_nvenc",  # NVIDIA hardware acceleration
            "-preset",
            preset,
            "-b:v",
            "20M",  # Use CRF for quality control
            "-c:a",
            "aac",
            output_path,
        ]
    )
    # check encoding of input video
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    ffprobe_output = subprocess.check_output(ffprobe_cmd, text=True)
    if "h264_cuvid" in ffprobe_output:
        print("Hardware accelerated encoding enabled")
    else:
        print(
            "Hardware accelerated encoding disabled, falling back to software encoding"
        )

    # Execute command and capture stdout and stderr
    try:
        if ffprobe_output.strip() == "h264_cuvid":
            subprocess.run(
                cmd,
                check=True,
            )
        else:
            raise ValueError("Hardware accelerated decoding not supported.")
    except Exception as e:
        if isinstance(e, subprocess.CalledProcessError):
            print(
                "Hardware accelerated decoding failed, falling back to normal decoding"
            )

        # If hardware acceleration fails, fall back to software encoding
        fallback_cmd = ["ffmpeg", "-y", "-loglevel", "error"]

        # Add trimming parameters
        if start_time != 0:
            fallback_cmd.extend(["-ss", str(start_time)])

        if end_time != -1:
            fallback_cmd.extend(["-to", str(end_time)])

        # Software encoding parameters
        fallback_cmd.extend(
            [
                "-i",
                input_path,
                "-c:v",
                "h264_nvenc",  # NVIDIA hardware acceleration
                "-preset",
                preset,
                "-b:v",
                "20M",  # Use CRF for quality control
                "-c:a",
                "aac",
                output_path,
            ]
        )
        try:
            subprocess.run(fallback_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                "Hardware accelerated encoding failed, falling back to software encoding."
            )
            print(f"Error details: {e.stderr.decode('utf-8')}")
            fallback_cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
            ]
            if start_time != 0:
                fallback_cmd.extend(["-ss", str(start_time)])

            if end_time != -1:
                fallback_cmd.extend(["-to", str(end_time)])

            # Software encoding parameters
            fallback_cmd.extend(
                [
                    "-i",
                    input_path,
                    "-preset",
                    preset,
                    "-b:v",
                    "20M",  # Use CRF for quality control
                    "-c:a",
                    "aac",
                    output_path,
                ]
            )
            try:
                subprocess.run(fallback_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Re-encoding failed: {e}")
                print(f"Error details: {e.stderr.decode('utf-8')}")
                raise e
    return sieve.File(path=output_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        input_video = sieve.File(path=sys.argv[1])
        start = float(sys.argv[2]) if len(sys.argv) > 2 else 0
        end = float(sys.argv[3]) if len(sys.argv) > 3 else -1
        result = reencoder(video=input_video, start_time=start, end_time=end)
        print(f"Re-encoded video saved to: {result.path}")
    else:
        print("Usage: python reencoder.py <video_path> [start_time] [end_time]")
