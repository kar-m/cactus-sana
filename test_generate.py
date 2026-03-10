import sys
import json
import ctypes
import numpy as np
from PIL import Image

sys.path.append('python/src')
import cactus

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_generate.py <model_dir> <prompt>")
        sys.exit(1)

    model_dir = sys.argv[1]
    prompt = sys.argv[2]
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    print(f"Loading model from {model_dir}...")
    model = cactus.cactus_init(model_dir)
    if not model:
        print(f"Failed to load model: {cactus.cactus_get_last_error()}")
        sys.exit(1)

    try:
        print(f"Generating image for prompt: '{prompt}'")
        res_json = cactus.cactus_generate_image(model, prompt, width=size, height=size)
        res = json.loads(res_json)
        
        if not res.get("success"):
            print(f"Generation failed: {res.get('error')}")
            return

        out_node = res["output_node"]
        width = res["width"]
        height = res["height"]
        print(f"Generation complete. Time: {res['total_time_ms']}ms. Node ID: {out_node}")

        # Retrieve the raw output buffer pointer
        ptr = cactus.cactus_get_output(model, out_node)
        if not ptr:
            print("Failed to get output pointer from graph.")
            return

        # The VAE outputs raw FP16 values in [N, C, H, W] format.
        # N=1, C=3, H=height, W=width.
        elements = 3 * height * width
        
        # Cast the raw pointer to a ctypes array of FP16 (uint16 representation)
        fp16_array_type = ctypes.c_uint16 * elements
        fp16_array = ctypes.cast(ptr, ctypes.POINTER(fp16_array_type)).contents
        
        # Convert to numpy array of float16, then float32 for processing
        np_fp16 = np.frombuffer(fp16_array, dtype=np.float16)
        img_arr = np_fp16.astype(np.float32)

        # Reshape to [C, H, W]
        img_arr = img_arr.reshape((3, height, width))
        
        # Convert to [H, W, C]
        img_arr = np.transpose(img_arr, (1, 2, 0))
        
        # The output is in [-1, 1], convert to [0, 255] RGB
        img_arr = (img_arr + 1.0) / 2.0
        img_arr = np.clip(img_arr, 0.0, 1.0) * 255.0
        img_arr = img_arr.astype(np.uint8)

        # Save the image
        img = Image.fromarray(img_arr)
        out_filename = "output.png"
        img.save(out_filename)
        print(f"Saved generated image to {out_filename}")

    finally:
        cactus.cactus_destroy(model)

if __name__ == "__main__":
    main()
