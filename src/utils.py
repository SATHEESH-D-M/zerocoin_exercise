import pyopencl as cl
import numpy as np
from PIL import Image
import os
import logging
from typing import Tuple, List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Enable full OpenCL compiler output for debugging
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def setup_logging():
    """Configure the root logger with both console and file handlers."""
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"application_{timestamp}.log")

    # Handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.INFO)

    # Format
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Configure root logger (Python 3.8+ can use force=True)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler],
        force=True,  # remove any existing handlers first[10][11]
    )

    logging.info("Logging initialized")
    logging.info(f"Log file: {log_filename}")
    return logging.getLogger()  # the root logger


def load_image(image_path: str) -> np.array:
    """
    Load an RGBA image and convert it to a NumPy array.

    Args:
        image_path (string): Path to the input image.

    Returns:
        img_np: NumPy array of the image in RGBA format.
    """
    try:
        logging.info(f"Loading image ...")
        img = Image.open(image_path)
        # Convert to RGBA if not already in that mode
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # Convert to NumPy array of uint8
        img_np = np.array(img, dtype=np.uint8)
        # Log the shape of the image
        logging.info(f"Image loaded with dimensions: {img_np.shape} (HxWxC)")
    except Exception as e:
        logging.error(f"Failed to load image from {image_path}: {e}")
        raise e

    return img_np


def setup_opencl() -> tuple[cl.Context, cl.CommandQueue]:
    """
    Set up OpenCL context and command queue with detailed try/except checks and logging.
    steps include:
    1. Discover platforms
    2. Select platform
    3. Discover devices
    4. Select device
    5. Verify image support
    6. Create context
    7. Validate context
    8. Create command queue
    9. Validate command queue

    Returns:
        (cl.Context, cl.CommandQueue): The created context and command queue.
    Raises:
        RuntimeError: If any step in initialization fails.
    """
    logging.info("Starting OpenCL setup...")
    # Initialize context and queue to None
    ctx = queue = None

    # 1. Discover platforms
    try:
        logging.info("Querying OpenCL platforms...")
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        logging.info(f"Found {len(platforms)} platform(s)")
    except Exception as e:
        logging.error(f"Failed to get OpenCL platforms: {e}")
        raise

    # 2. Select platform
    try:
        platform = platforms[0]
        logging.info(f"Using platform: {platform.name}")
    except Exception as e:
        logging.error(f"Failed to select OpenCL platform: {e}")
        raise

    # 3. Discover devices
    try:
        logging.info(f"Querying devices on platform '{platform.name}'...")
        devices = platform.get_devices()
        if not devices:
            raise RuntimeError("No OpenCL devices found on selected platform")
        logging.info(f"Found {len(devices)} device(s) on platform")
    except Exception as e:
        logging.error(f"Failed to get devices: {e}")
        raise

    # 4. Select device
    try:
        device = devices[0]
        logging.info(f"Using device: {device.name}")
    except Exception as e:
        logging.error(f"Failed to select OpenCL device: {e}")
        raise

    # 5. Verify image support
    try:
        logging.info("Checking image support on device...")
        if not device.image_support:
            raise RuntimeError(
                "Selected device does not support image operations"
            )
        logging.info("Device supports images")
    except Exception as e:
        logging.error(f"Image support check failed: {e}")
        raise

    # 6. Create context
    try:
        logging.info("Creating OpenCL context...")
        ctx = cl.Context([device])
        logging.info("OpenCL context created")
    except Exception as e:
        logging.error(f"Failed to create OpenCL context: {e}")
        raise

    # 7. (Context validity) – implicit in creation; no cl.context_is_valid()
    # If additional validation is desired, query context_info:
    try:
        devs_in_ctx = ctx.get_info(cl.context_info.DEVICES)
        if device not in devs_in_ctx:
            raise RuntimeError("Context does not include the selected device")
        logging.info("Context validation passed")
    except Exception as e:
        logging.error(f"Context validation failed: {e}")
        raise

    # 8. Create command queue
    try:
        logging.info("Creating command queue with profiling enabled...")
        queue = cl.CommandQueue(
            ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
        )
        logging.info("Command queue created")
    except Exception as e:
        logging.error(f"Failed to create command queue: {e}")
        raise

    # 9. (Queue validity) – test simple operation
    try:
        logging.info("Validating command queue by enqueuing a no-op...")
        queue.finish()  # This will raise if the queue is invalid
        logging.info("Command queue validation passed")
    except Exception as e:
        logging.error(f"Command queue validation failed: {e}")
        raise

    logging.info("OpenCL setup completed successfully")

    return ctx, queue


def opencl_image_format(
    channel_order=cl.channel_order.RGBA,
    channel_type=cl.channel_type.UNORM_INT8,
) -> cl.ImageFormat:
    """
    Define the OpenCL image format.

    Returns:
        cl.ImageFormat: The image format for RGBA images by default.
    """
    logging.info(
        f"Defining OpenCL image format: {channel_order}, {channel_type}"
    )
    return cl.ImageFormat(channel_order, channel_type)


def upload_image_to_gpu(
    ctx: cl.Context,
    img_np: np.ndarray,
    image_format: cl.ImageFormat,
    mem_flags: cl.mem_flags = cl.mem_flags.READ_ONLY
    | cl.mem_flags.COPY_HOST_PTR,
) -> Tuple[cl.Image, Tuple[int, int]]:
    """
    Upload an RGBA NumPy image to the GPU as an OpenCL 2D image,
    using the outputs of load_image() and opencl_image_format() directly.

    Args:
        ctx:            A validated OpenCL context.
        img_np:         NumPy array of shape (H, W, 4), dtype=uint8,
                        as returned by load_image().
        image_format:   cl.ImageFormat instance as returned by opencl_image_format().
        mem_flags:      OpenCL memory flags (default: READ_ONLY | COPY_HOST_PTR).

    Returns:
        gpu_image:      The cl.Image object on the GPU.
        (width, height): The image dimensions.
    """
    logging.info("Uploading image to GPU...")

    # img_np is assumed to be RGBA uint8 and correctly formatted
    height, width, channels = img_np.shape
    if channels != 4:
        logging.warning(
            f"Expected 4 channels, got {channels}. Proceeding anyway."
        )

    # Create the OpenCL image on the GPU using the provided image_format
    gpu_image = cl.Image(
        ctx,
        mem_flags,
        format=image_format,
        shape=(width, height),
        hostbuf=img_np,
    )

    logging.info(
        f"Uploaded image to GPU: {width}x{height} pixels, format={image_format}"
    )
    return gpu_image, (width, height)


def gpu_buffer(
    ctx: cl.Context,
    shape: Tuple[int, int],
    fmt: cl.ImageFormat,
    flags: cl.mem_flags = cl.mem_flags.READ_WRITE,
) -> cl.Image:
    """
    Create a 2D image in GPU memory with simple error handling and logging.

    Args:
        ctx:    OpenCL context
        shape:  (width, height)
        fmt:    cl.ImageFormat
        flags:  combination of READ_ONLY/WRITE_ONLY/READ_WRITE

    Returns:
        cl.Image: The created GPU image

    Raises:
        RuntimeError: If image creation fails
    """
    width, height = shape
    try:
        logging.info(
            f"Creating GPU buffer: size=({width}x{height}), format={fmt}, flags={flags}"
        )
        image = cl.Image(ctx, flags, fmt, shape=(width, height))
        logging.info("GPU buffer created successfully")
        return image

    except cl.Error as e:
        logging.error(f"Failed to create GPU buffer: {e}")
        raise RuntimeError(
            f"gpu_buffer error: could not create image2d_t"
        ) from e


def build_program(
    ctx: cl.Context, src_path: str, required_kernels: List[str]
) -> Tuple[cl.Program, Dict[str, cl.Kernel]]:
    """
    Load, compile OpenCL source, and extract kernels, with error handling and logging.

    Args:
        ctx:               OpenCL context
        src_path:          Path to .cl file
        required_kernels:  List of kernel names to fetch

    Returns:
        (program, kernels) where kernels[name] is cl.Kernel

    Raises:
        RuntimeError: On file I/O, build errors, or missing kernels.
    """
    logging.info(f"Building OpenCL program from '{src_path}'")
    # 1. Read source file
    try:
        source = Path(src_path).read_text()
        logging.info("Kernel source loaded successfully")
    except FileNotFoundError as fnf:
        logging.error(f"Kernel file not found: {src_path}")
        raise RuntimeError(f"Could not find .cl file at '{src_path}'") from fnf
    except Exception as e:
        logging.error(f"Error reading kernel file '{src_path}': {e}")
        raise RuntimeError(f"Failed to read kernel source '{src_path}'") from e

    # 2. Build program
    try:
        program = cl.Program(ctx, source).build()
        kernel_names = program.get_info(cl.program_info.KERNEL_NAMES)
        logging.info(
            f"OpenCL program built successfully; exported kernels: {kernel_names}"
        )
    except cl.RuntimeError as build_err:
        logging.error(f"OpenCL build failed: {build_err}")
        # Attempt to retrieve and log the build log
        try:
            # ctx.devices may list devices used to build
            for dev in ctx.devices:
                log = program.get_build_info(dev, cl.program_build_info.LOG)
                logging.error(f"Build log for device {dev.name}:\n{log}")
        except Exception:
            logging.error("Unable to retrieve detailed build log")
        raise RuntimeError("OpenCL program build failed") from build_err

    # 3. Extract kernels
    kernels: Dict[str, cl.Kernel] = {}
    for name in required_kernels:
        try:
            kernels[name] = cl.Kernel(program, name)
            logging.info(f"Kernel '{name}' loaded")
        except cl.LogicError as e:
            logging.error(f"Kernel '{name}' not found in program")
            raise RuntimeError(
                f"Kernel '{name}' not found in compiled program"
            ) from e

    logging.info("All requested kernels loaded successfully")
    return program, kernels


def execute_kernels_dynamic(
    program: cl.Program,
    kernels: Dict[str, cl.Kernel],
    queue: cl.CommandQueue,
    ctx: cl.Context,
    input_image: cl.Image,
    image_format: cl.ImageFormat,
    max_luminance: float,
    global_size: Tuple[int, int],
    local_size: Tuple[int, int] = None,
) -> List[cl.Image]:
    """
    Execute multiple kernels dynamically with automatic buffer management and synchronization.

    Args:
        program: Built OpenCL program
        kernels: Dictionary of kernel objects from build_program
        queue: OpenCL command queue
        ctx: OpenCL context
        input_image: Initial input image
        image_format: Image format for buffer creation
        max_luminance: Parameter for tone mapping
        global_size: (width, height) work size
        local_size: Optional work-group size

    Returns:
        List of all image buffers with final output first
    """
    # Create buffer sequence: input + intermediate buffers + final output
    buffers = [input_image]
    kernel_names = list(kernels.keys())

    # Create intermediate and output buffers
    for i in range(len(kernel_names)):
        flags = (
            cl.mem_flags.READ_WRITE
            if i < len(kernel_names) - 1
            else cl.mem_flags.WRITE_ONLY
        )
        buffers.append(gpu_buffer(ctx, global_size, image_format, flags))

    events = []
    prev_event = None

    # Execute kernels in sequence with synchronization
    for i, kernel_name in enumerate(kernel_names):
        logging.info(
            f"Executing kernel {i + 1}/{len(kernel_names)}: {kernel_name}"
        )

        # Prepare kernel arguments
        kernel = kernels[kernel_name]
        args = [buffers[i], buffers[i + 1]]

        # Add kernel-specific parameters
        if kernel_name == "tone_map_logarithmic":
            args.append(np.float32(max_luminance))

        # Execute kernel with dependency on previous event
        kernel_event = kernel(
            queue,
            global_size,
            local_size,
            *args,
            wait_for=[prev_event] if prev_event else [],
        )

        events.append(kernel_event)
        prev_event = kernel_event

    # Wait for final kernel to complete
    prev_event.wait()

    # Return buffers with final output first, then intermediates, then input
    return [buffers[-1]] + buffers[1:-1] + [buffers[0]]


def download_save_display(
    queue: cl.CommandQueue,
    final_image: cl.Image,
    img_np: np.ndarray,
    output_path: str,
    max_luminance: float,
) -> np.ndarray:
    """
    Download final image from GPU, save to disk, display comparison,
    and handle cleanup with error handling and logging.

    Args:
        queue: OpenCL command queue
        final_image: GPU image with final processed data
        img_np: Original CPU image array (for comparison)
        output_path: Path to save processed image
        max_luminance: Tone mapping parameter for display title

    Returns:
        result_np: Processed image as NumPy array

    Raises:
        RuntimeError: If any step fails
    """
    # Validate inputs
    if not isinstance(img_np, np.ndarray) or img_np.dtype != np.uint8:
        logging.error("Invalid img_np: must be uint8 NumPy array")
        raise ValueError("img_np must be uint8 NumPy array")

    try:
        # === 1. Download image from GPU ===
        logging.info("Copying results back to CPU...")
        result_np = np.empty_like(img_np)
        height, width = img_np.shape[:2]

        # Define copy region (width, height, depth)
        origin = (0, 0, 0)
        region = (width, height, 1)

        # Blocking copy to ensure completion
        cl.enqueue_copy(
            queue,
            result_np,
            final_image,
            origin=origin,
            region=region,
            is_blocking=True,
        )
        logging.info(f"Downloaded image: {width}x{height} pixels")

        # === 2. Save processed image ===
        logging.info(f"Saving processed image to {output_path}")
        try:
            Image.fromarray(result_np, "RGBA").save(output_path)
            logging.info("Image saved successfully")
        except IOError as e:
            logging.error(f"Failed to save image: {e}")
            raise

        # === 3. Create and save comparison ===
        logging.info("Creating image comparison...")
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Original image
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Processed image
            axes[1].imshow(result_np)
            axes[1].set_title(f"Processed (L_max={max_luminance})")
            axes[1].axis("off")

            plt.tight_layout()

            # Save comparison figure
            comp_path = output_path.replace(".png", "_comparison.png")
            plt.savefig(comp_path, dpi=150, bbox_inches="tight")
            logging.info(f"Saved comparison to {comp_path}")

            plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            logging.warning(f"Image display failed: {e}")
            # Continue even if display fails

        logging.info("Processing complete!")
        return result_np

    except cl.LogicError as e:
        logging.error(f"CL logic error during download: {e}")
        raise RuntimeError("Image download failed") from e
    except cl.RuntimeError as e:
        logging.error(f"CL runtime error during download: {e}")
        raise RuntimeError("Image download failed") from e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
