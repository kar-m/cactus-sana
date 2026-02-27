#!/usr/bin/env python3
"""Generate images with Sana via Cactus FFI (text2img or img2img)."""

import argparse
import ctypes
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python", "src"))

from cactus import (
    cactus_destroy,
    cactus_generate_image,
    cactus_generate_image_to_image,
    cactus_get_output,
    cactus_init,
)


def save_ppm(path: str, rgb_hwc: np.ndarray) -> None:
    h, w, c = rgb_hwc.shape
    if c != 3:
        raise ValueError("Expected RGB image")
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb_hwc.tobytes())


def decode_output_rgb(model, node_id: int, width: int, height: int) -> np.ndarray:
    ptr = cactus_get_output(model, node_id)
    if not ptr:
        raise RuntimeError("cactus_get_output returned null pointer")

    total = 3 * width * height
    raw = (ctypes.c_uint16 * total).from_address(ptr)
    fp16 = np.frombuffer(raw, dtype=np.uint16).view(np.float16)
    chw = fp16.astype(np.float32).reshape(3, height, width)

    # Sana decoder output is approximately in [-1, 1].
    hwc = ((chw.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return hwc


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image with Sana via Cactus")
    parser.add_argument("--model", default="./weights/sana-0.6b", help="Model weights directory")
    parser.add_argument("--prompt", default="A cinematic portrait of a red fox in snowfall", help="Text prompt")
    parser.add_argument("--width", type=int, default=1024, help="Output width (must match model graph)")
    parser.add_argument("--height", type=int, default=1024, help="Output height (must match model graph)")
    parser.add_argument("--output", default="sana_output.ppm", help="Output path (PPM)")
    parser.add_argument("--init-image", default=None, help="Optional init image path for img2img")
    parser.add_argument("--strength", type=float, default=0.6, help="Img2img strength [0, 1]")
    args = parser.parse_args()

    model = cactus_init(args.model)
    if not model:
        raise RuntimeError("Failed to initialize model")

    try:
        if args.init_image:
            result_json = cactus_generate_image_to_image(
                model,
                args.prompt,
                args.init_image,
                width=args.width,
                height=args.height,
                strength=args.strength,
            )
        else:
            result_json = cactus_generate_image(
                model,
                args.prompt,
                width=args.width,
                height=args.height,
            )

        result = json.loads(result_json)
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Unknown Sana generation error"))

        node_id = int(result["output_node"])
        image = decode_output_rgb(model, node_id, args.width, args.height)
        save_ppm(args.output, image)

        print(json.dumps(result, indent=2))
        print(f"Saved image to {args.output}")
    finally:
        cactus_destroy(model)


if __name__ == "__main__":
    main()
