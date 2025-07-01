/* gaussian_blur_and_tone_mapping.cl
 *
 * Two-stage image pipeline:
 *   1. 3×3 Gaussian blur applied per channel (R, G, B).
 *   2. Logarithmic tone mapping on luminance, preserving colour ratios.
 *
 * All math is done in normalized float [0,1] space; the alpha channel
 * is passed through untouched.
 *
 * Copyright © 2025  (MIT-licensed — free to reuse).
 */

/* ------------------------------------------------------------------------- */
/*  Sampler shared by both kernels                                            */
/* ------------------------------------------------------------------------- */
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP          |
                               CLK_FILTER_NEAREST;

/* ------------------------------------------------------------------------- */
/*  3×3 Gaussian kernel (sum = 16).                                           */
/*  Each entry = original weight ÷ 16 so the kernel is already normalised.    */
/* ------------------------------------------------------------------------- */
__constant float g_kernel[3][3] = {
    { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f },
    { 2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f },
    { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f }
};

/* ------------------------------------------------------------------------- */
/*  Kernel 1 : gaussian_blur                                                 */
/* ------------------------------------------------------------------------- */
__kernel void gaussian_blur(__read_only  image2d_t input_image,
                            __write_only image2d_t output_image)
{
    /* Absolute pixel coordinate we are responsible for */
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    /* Accumulator for RGB (alpha handled separately) */
    float3 rgb_sum = (float3)(0.0f);

    /* Loop over 3×3 neighbourhood */
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int2 offset = (int2)(coord.x + dx, coord.y + dy);

            /* Sample neighbour with address-clamp mode           */
            /* Alpha ignored for blur (copied later unchanged)    */
            float4 neighbour = read_imagef(input_image, sampler, offset);

            /* Weighted accumulation (xyz only)                   */
            float weight = g_kernel[dy + 1][dx + 1];
            rgb_sum += weight * neighbour.xyz;
        }
    }

    /* Fetch alpha from the exact source pixel           */
    float alpha = read_imagef(input_image, sampler, coord).w;

    /* Re-assemble final blurred pixel                    */
    float4 blurred_pixel = (float4)(rgb_sum.x, rgb_sum.y, rgb_sum.z, alpha);

    /* Store back to output image                         */
    write_imagef(output_image, coord, blurred_pixel);
}

/* ------------------------------------------------------------------------- */
/*  Kernel 2 : tone_map_logarithmic                                          */
/* ------------------------------------------------------------------------- */
__kernel void tone_map_logarithmic(__read_only  image2d_t input_image,
                                   __write_only image2d_t output_image,
                                   const float  max_luminance)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    /* Read blurred pixel */
    float4 pix = read_imagef(input_image, sampler, coord);
    float3 rgb = pix.xyz;
    float  a   = pix.w;

    /* Compute luminance Y using Rec. 709 constants */
    float Y = dot(rgb, (float3)(0.2126f, 0.7152f, 0.0722f));

    /* Tone-map the luminance with log compression       */
    float Y_out = (Y > 0.0f) ? log(1.0f + Y) / log(1.0f + max_luminance)
                             : 0.0f;

    /* Re-scale RGB so that luminance matches Y_out       */
    float3 rgb_out = (Y > 0.0f) ? (Y_out / Y) * rgb
                                : (float3)(0.0f);

    /* Clamp final colour to [0,1] just in case           */
    rgb_out = clamp(rgb_out, 0.0f, 1.0f);

    /* Re-assemble, preserving alpha                      */
    float4 out_pixel = (float4)(rgb_out.x, rgb_out.y, rgb_out.z, a);

    write_imagef(output_image, coord, out_pixel);
}
