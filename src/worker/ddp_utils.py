import os
import platform
import tempfile

def get_ddp_init_method():
    """
    Returns the recommended init_method for PyTorch DDP based on platform.
    - On Windows, uses a file:// rendezvous to avoid libuv issues.
    - On Linux/macOS, uses TCP if available.
    """
    system = platform.system()

    if system == "Windows":
        # Use temporary file rendezvous
        temp_dir = tempfile.gettempdir()
        init_file = os.path.join(temp_dir, "ddp_init")
        return f"file://{init_file}"
    else:
        # Use TCP for Linux/macOS (assuming port 12355 is free)
        return "tcp://127.0.0.1:12355"
