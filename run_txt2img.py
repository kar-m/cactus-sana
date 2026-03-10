#!/usr/bin/env python3
import os, sys, json, ctypes
import numpy as np
from PIL import Image

sys.path.append('python/src')
import cactus

MODEL = "./weights/sana-0.6b"
SIZE  = 1024
OUT   = "results"

os.makedirs(OUT, exist_ok=True)
os.environ["CACTUS_SANA_IMAGE_SIZE"] = str(SIZE)
os.environ["CACTUS_SANA_SEED"]       = "42"

PROMPTS = [
    ("t2i_01_astronaut",   "an astronaut floating in space, Earth visible behind, photorealistic"),
    ("t2i_02_forest",      "a misty forest at dawn, sunbeams through ancient trees, cinematic"),
    ("t2i_03_city",        "futuristic Tokyo skyline at night, neon reflections on wet streets"),
    ("t2i_04_portrait",    "portrait of a woman with braided hair, golden hour light, film photography"),
    ("t2i_05_cat",         "a fluffy orange cat sitting on a windowsill, soft afternoon light"),
    ("t2i_06_mountain",    "dramatic mountain landscape at sunset, snow-capped peaks, golden sky"),
    ("t2i_07_food",        "a bowl of ramen with soft boiled egg, steam rising, restaurant lighting"),
    ("t2i_08_abstract",    "abstract fluid art, swirling blues and golds, marble texture"),
]

model = cactus.cactus_init(MODEL)
if not model:
    print("Failed:", cactus.cactus_get_last_error()); sys.exit(1)

try:
    for name, prompt in PROMPTS:
        print(f"\n[{name}]\n  '{prompt}'")
        res_json = cactus.cactus_generate_image(model, prompt, width=SIZE, height=SIZE)
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
