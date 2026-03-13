#!/usr/bin/env python3
"""Test img2img pipeline for Sana model.

Usage:
    python test_img2img.py <model_dir> <input_image> <prompt> [strength] [size]

Example:
    CACTUS_SANA_SEED=42 CACTUS_SANA_IMAGE_SIZE=512 \
        python test_img2img.py ./weights/sana-0.6b output.png "a blue apple on white background" 0.7 512
"""
import sys
import json
import ctypes
import numpy as np
from PIL import Image

sys.path.append('python/src')
import cactus


def main():
    if len(sys.argv) < 4:
        print("Usage: python test_img2img.py <model_dir> <input_image> <prompt> [strength=0.7] [size=512]")
        sys.exit(1)

    model_dir    = sys.argv[1]
    input_image  = sys.argv[2]
    prompt       = sys.argv[3]
    strength     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
    size         = int(sys.argv[5])   if len(sys.argv) > 5 else 512

    print(f"Loading model from {model_dir}...")
    model = cactus.cactus_init(model_dir)
    if not model:
        print(f"Failed to load model: {cactus.cactus_get_last_error()}")
        sys.exit(1)

    try:
        print(f"Running img2img: '{prompt}' (strength={strength}) on {input_image}")
        res_json = cactus.cactus_generate_image_to_image(
            model, prompt, input_image,
            width=size, height=size, strength=strength
        )
        res = json.loads(res_json)

        if not res.get("success"):
            print(f"Generation failed: {res.get('error')}")
            return

        out_node = res["output_node"]
        width    = res["width"]
        height   = res["height"]
        print(f"Done. Time: {res['total_time_ms']:.0f}ms")

        ptr = cactus.cactus_get_output(model, out_node)
        if not ptr:
            print("Failed to get output pointer.")
            return

        elements = 3 * height * width
        fp16_array = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16 * elements)).contents
        img_arr = np.frombuffer(fp16_array, dtype=np.float16).astype(np.float32)
        img_arr = img_arr.reshape((3, height, width))
        img_arr = np.transpose(img_arr, (1, 2, 0))
        img_arr = np.clip((img_arr + 1.0) / 2.0, 0.0, 1.0) * 255.0
        img_arr = img_arr.astype(np.uint8)

        out_path = "output_img2img.png"
        Image.fromarray(img_arr).save(out_path)
        print(f"Saved to {out_path}")

    finally:
        cactus.cactus_destroy(model)


if __name__ == "__main__":
    main()
