import subprocess
import os
import logging
import importlib

logger = logging.getLogger(__name__)

# Dynamically initialize static-ffmpeg if available without IDE unresolved import errors
try:
    static_ffmpeg = importlib.import_module("static_ffmpeg")
    if hasattr(static_ffmpeg, "add_paths"):
        static_ffmpeg.add_paths()
        logger.info("static-ffmpeg initialized and added to PATH.")
except (ImportError, ModuleNotFoundError):
    logger.info("static_ffmpeg not present in active environment; relying on system ffmpeg.")
except Exception as e:
    logger.warning(f"Notice: static-ffmpeg initialization skipped: {e}")

def convert_to_wav(input_path: str, output_path: str) -> bool:
    """
    Converts input audio file to 16kHz mono WAV format using FFmpeg.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        return False

    try:
        # Build the FFmpeg command
        # -y: Overwrite output files without asking
        # -i: Input file path
        # -ar 16000: Set audio sampling rate to 16000 Hz
        # -ac 1: Set audio channels to 1 (mono)
        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]
        
        logger.info(f"Running command: {' '.join(command)}")
        # Execute the conversion
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("Audio conversion successful.")
        return True
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'No stderr'
        stdout_output = e.stdout.decode('utf-8', errors='ignore') if e.stdout else 'No stdout'
        logger.error(f"FFmpeg conversion failed. Stderr: {stderr_output}. Stdout: {stdout_output}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during audio conversion: {e}")
        return False
