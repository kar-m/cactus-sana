#!/usr/bin/env python3
import os, sys, json, ctypes
import numpy as np
from PIL import Image

sys.path.append('python/src')
import cactus

MODEL = "./weights/sana-1.0-0.6b-512-int8"
INPUT = "apple.jpg"
SIZE  = 512
OUT   = "results"

os.makedirs(OUT, exist_ok=True)
os.environ["CACTUS_SANA_IMAGE_SIZE"] = str(SIZE)
os.environ["CACTUS_SANA_STEPS"]      = "10"
os.environ["CACTUS_SANA_SEED"]       = "42"

EXAMPLES = [
    ("01_blue_apple",       "a blue apple on white background",                    0.9),
    ("02_golden_apple",     "a golden metallic apple on white background",         0.9),
    ("03_watercolor",       "a watercolor painting of an apple",                   0.7),
    ("04_sketch",           "a pencil sketch of an apple, black and white",        0.7),
    ("05_glass_apple",      "a transparent glass apple on white background",       0.85),
    ("06_strawberry",       "a red strawberry on white background",                0.85),
    ("07_lemon",            "a yellow lemon on white background",                  0.9),
    ("08_pumpkin",          "a small orange pumpkin on white background",          0.9),
    ("09_neon",             "a glowing neon apple, dark background, cyberpunk",    0.95),
    ("10_oil_painting",     "an oil painting of an apple, impressionist style",    0.7),
]

model = cactus.cactus_init(MODEL)
if not model:
    print("Failed:", cactus.cactus_get_last_error()); sys.exit(1)

try:
    for name, prompt, strength in EXAMPLES:
        print(f"\n[{name}] strength={strength}  '{prompt}'")
        res_json = cactus.cactus_generate_image_to_image(
            model, prompt, INPUT, width=SIZE, height=SIZE, strength=strength)
        res = json.loads(res_json)
        if not res.get("success"):
            print(f"  FAILED: {res.get('error')}"); continue

        out_node = res["output_node"]
        w, h = res["width"], res["height"]
        ptr = cactus.cactus_get_output(model, out_node)
        fp16 = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16 * (3*h*w))).contents
        arr  = np.frombuffer(fp16, dtype=np.float16).astype(np.float32)
        arr  = arr.reshape(3, h, w).transpose(1, 2, 0)
        arr  = np.clip((arr + 1.0) / 2.0, 0.0, 1.0) * 255
        out_path = f"{OUT}/{name}.png"
        Image.fromarray(arr.astype(np.uint8)).save(out_path)
        print(f"  {res['total_time_ms']:.0f}ms → {out_path}")
finally:
    cactus.cactus_destroy(model)

print("\nDone.")
