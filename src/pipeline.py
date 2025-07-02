"""
This module implements a complete OpenCL image processing pipeline
non reusable code specific for this pipeline.
functions include:
- process_image
"""

from utils import *
import argparse
import logging


def process_image(
    image_path: str, output_path: str, max_luminance: float = 5.0
) -> np.ndarray:
    """
    Complete OpenCL image processing pipeline with proper synchronization.

    Args:
        image_path: Path to input image
        output_path: Path for output image
        max_luminance: Tone mapping parameter (default: 5.0)

    Returns:
        result_np: Processed image as NumPy array
    """
    try:
        # 1. Load image
        img_np = load_image(image_path)

        # 2. Set up OpenCL
        ctx, queue = setup_opencl()

        # 3. Define image format
        image_format = opencl_image_format()  # Uses RGBA/UNORM_INT8 by default

        # 4. Upload input image to GPU
        input_image, (width, height) = upload_image_to_gpu(
            ctx, img_np, image_format
        )
        global_size = (width, height)

        # 5. Build OpenCL program
        kernel_file = "src/gaussian_blur_and_tone_mapping.cl"
        required_kernels = ["gaussian_blur", "tone_map_logarithmic"]
        program, kernels = build_program(ctx, kernel_file, required_kernels)

        # 6. Execute kernels dynamically
        image_buffers = execute_kernels_dynamic(
            program=program,
            kernels=kernels,
            queue=queue,
            ctx=ctx,
            input_image=input_image,
            image_format=image_format,
            max_luminance=max_luminance,
            global_size=global_size,
        )

        # The first buffer in the list is the final output
        final_image = image_buffers[0]

        # 7. Download, save, and display results
        result_np = download_save(
            queue=queue,
            final_image=final_image,
            img_np=img_np,
            output_path=output_path,
            max_luminance=max_luminance,
        )

        return result_np

    except Exception as e:
        logging.error(f"Image processing pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Initialize logging
    logger = setup_logging()

    try:
        # Argument parser setup
        parser = argparse.ArgumentParser(
            description="OpenCL Image Processing Pipeline"
        )
        parser.add_argument(
            "--input",
            type=str,
            help="Path to input image (default: sample.png)",
            default="/Users/satheeshdm/Desktop/MTech/Semester-3/Project/zerocoin_application/zerocoin_exercise/data/sample.png",
        )
        parser.add_argument(
            "--output",
            type=str,
            help="Path for output image (default: data/output.png)",
            default="data/output.png",
        )
        parser.add_argument(
            "--mLuminance",
            type=float,
            help="max luminance for tone mapping (default: 5.0)",
            default=0.5,
        )
        args = parser.parse_args()

        logger.info(f"Processing image: {args.input}")

        # Call your processing function
        process_image(args.input, args.output, args.mLuminance)

    except Exception as e:
        logger.exception("An error occurred during processing.")
