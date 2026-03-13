#!/usr/bin/env python3
"""
Convert Sana-Sprint HuggingFace weights to Cactus binary format.

Usage:
    python convert_sana.py --output ./weights/sana-0.6b

This downloads/loads the three Sana sub-models from HuggingFace:
  1. Text encoder (Gemma2 2B) -> output/text_encoder/
  2. Transformer denoiser (SanaTransformer2DModel) -> output/ (root)
  3. VAE (AutoencoderDC) -> output/vae/

Prerequisites:
    pip install torch diffusers transformers accelerate
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent to path so we can import cactus modules
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from src.tensor_io import save_tensor_with_header, create_quantization_stats, print_quantization_summary

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def save(tensor, path, precision="FP16"):
    """Save a tensor as a .weights file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = create_quantization_stats()
    save_tensor_with_header(tensor, path, precision=precision, stats_tracker=stats)


def write_config(path, config_dict):
    """Write a config.txt key=value file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for k, v in config_dict.items():
            if isinstance(v, bool):
                f.write(f"{k}={'true' if v else 'false'}\n")
            elif isinstance(v, (list, tuple)):
                f.write(f"{k}={','.join(str(x) for x in v)}\n")
            else:
                f.write(f"{k}={v}\n")


def load_diffusers_component(model_cls, model_id, subfolder, variant: Optional[str] = None):
    """Load a diffusers component with optional variant and simple fallback variants."""
    attempts = []
    if variant is not None:
        attempts.append(variant)
    attempts.extend([None, "fp16", "bf16"])
    # Deduplicate while preserving order.
    seen = set()
    attempts = [a for a in attempts if not (a in seen or seen.add(a))]
    last_error = None
    for candidate in attempts:
        kwargs = dict(subfolder=subfolder, torch_dtype=torch.float32)
        if candidate is not None:
            kwargs["variant"] = candidate
        try:
            model = model_cls.from_pretrained(model_id, **kwargs)
            return model, candidate
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to load {model_cls.__name__} from {model_id}/{subfolder} "
        f"with variants {attempts}: {last_error}"
    )


# ──────────────────────────────────────────────
# 1. Text Encoder (Gemma2)
# ──────────────────────────────────────────────
def convert_text_encoder(output_dir, precision="INT8", model_id="google/gemma-2-2b-it"):
    """Convert Gemma2 text encoder using the existing cactus converter."""
    from src.converter import convert_hf_model_weights
    from src.tensor_io import format_config_value
    from src.tokenizer import convert_hf_tokenizer

    print(f"\n{'='*60}")
    print(f"Converting text encoder: {model_id}")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, token=hf_token)
    
    te_dir = Path(output_dir) / "text_encoder"
    te_dir.mkdir(parents=True, exist_ok=True)

    model_config = convert_hf_model_weights(model, te_dir, precision=precision)

    config_path = te_dir / "config.txt"
    with open(config_path, "w") as f:
        for key, value in model_config.items():
            f.write(f"{key}={format_config_value(value)}\n")

    # Export tokenizer assets expected by Cactus (vocab/merges/tokenizer_config/chat template).
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    convert_hf_tokenizer(tokenizer, te_dir, token=hf_token)

    print(f"Text encoder saved to {te_dir}")
    del tokenizer
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ──────────────────────────────────────────────
# 2. Transformer Denoiser (SanaTransformer2DModel)
# ──────────────────────────────────────────────
def convert_transformer(
    output_dir,
    precision="FP16",
    model_id="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    variant: Optional[str] = None,
):
    """Convert the SanaTransformer2DModel weights.

    Returns a dict with transformer config values needed by Cactus.
    """
    print(f"\n{'='*60}")
    print(f"Converting transformer denoiser")
    print(f"{'='*60}")

    from diffusers import SanaTransformer2DModel
    transformer, used_variant = load_diffusers_component(
        SanaTransformer2DModel,
        model_id=model_id,
        subfolder="transformer",
        variant=variant,
    )
    print(f"  Loaded transformer with variant={used_variant if used_variant else 'default'}")
    sd = transformer.state_dict()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Timestep embedder (t_embedder.mlp.*) ----
    # HF: time_embed.emb.timestep_embedder.linear_1.{weight,bias}
    #     time_embed.emb.timestep_embedder.linear_2.{weight,bias}
    # Cactus: t_embedder.mlp.{0,2}.{weight,bias}
    global_map = {
        # Timestep embedder (Sprint/guidance_embeds model: no .emb. in path)
        "time_embed.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
        "time_embed.timestep_embedder.linear_1.bias":   "t_embedder.mlp.0.bias",
        "time_embed.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
        "time_embed.timestep_embedder.linear_2.bias":   "t_embedder.mlp.2.bias",
        # Sana 1.0 (non-Sprint): time_embed.emb.timestep_embedder.*
        "time_embed.emb.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
        "time_embed.emb.timestep_embedder.linear_1.bias":   "t_embedder.mlp.0.bias",
        "time_embed.emb.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
        "time_embed.emb.timestep_embedder.linear_2.bias":   "t_embedder.mlp.2.bias",
        # Guidance embedder (present when guidance_embeds=True, i.e. Sana Sprint)
        "time_embed.guidance_embedder.linear_1.weight": "t_embedder.guidance_mlp.0.weight",
        "time_embed.guidance_embedder.linear_1.bias":   "t_embedder.guidance_mlp.0.bias",
        "time_embed.guidance_embedder.linear_2.weight": "t_embedder.guidance_mlp.2.weight",
        "time_embed.guidance_embedder.linear_2.bias":   "t_embedder.guidance_mlp.2.bias",
        # Additional timestep components
        "time_embed.linear.weight": "t_embedder.linear.weight",
        "time_embed.linear.bias":   "t_embedder.linear.bias",
        # Caption projection (text -> cross-attn dim)
        "caption_projection.linear_1.weight": "caption_projection.linear_1.weight",
        "caption_projection.linear_1.bias":   "caption_projection.linear_1.bias",
        "caption_projection.linear_2.weight": "caption_projection.linear_2.weight",
        "caption_projection.linear_2.bias":   "caption_projection.linear_2.bias",
        # Caption norm
        "caption_norm.weight": "caption_norm.weight",
        # Output norm
        "norm_out.norm.weight": "norm_out.norm.weight",
        "norm_out.norm.bias":   "norm_out.norm.bias",
    }
    for hf_name, cactus_name in global_map.items():
        if hf_name in sd:
            save(sd[hf_name], out / f"{cactus_name}.weights", precision)
            print(f"  {hf_name} -> {cactus_name}")

    # ---- Patch embedding (pos_embed.proj.*) ----
    # HF: patch_embed.proj.{weight,bias}
    for suffix in ["weight", "bias"]:
        hf_key = f"patch_embed.proj.{suffix}"
        cactus_key = f"pos_embed.proj.{suffix}"
        if hf_key in sd:
            save(sd[hf_key], out / f"{cactus_key}.weights", precision)
            print(f"  {hf_key} -> {cactus_key}")

    # ---- Transformer blocks ----
    # Find number of blocks
    block_indices = set()
    for k in sd.keys():
        if k.startswith("transformer_blocks."):
            idx = int(k.split(".")[1])
            block_indices.add(idx)
    num_blocks = max(block_indices) + 1 if block_indices else 0
    print(f"  Found {num_blocks} transformer blocks")

    qk_norm_exported = 0
    for i in range(num_blocks):
        hf_prefix = f"transformer_blocks.{i}"
        cactus_prefix = f"blocks.{i}"

        # Per-block weights
        block_map = {
            # scale_shift_table
            "scale_shift_table": "scale_shift_table",
            # norm1, norm2 (LayerNorm)
            "norm1.weight": "norm1.weight",
            "norm1.bias":   "norm1.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias":   "norm2.bias",
            # attn1 (linear self-attention)
            "attn1.to_q.weight": "attn1.to_q.weight",
            "attn1.to_k.weight": "attn1.to_k.weight",
            "attn1.to_v.weight": "attn1.to_v.weight",
            "attn1.norm_q.weight": "attn1.norm_q.weight",
            "attn1.norm_k.weight": "attn1.norm_k.weight",
            "attn1.to_out.0.weight": "attn1.to_out.0.weight",
            "attn1.to_out.0.bias": "attn1.to_out.0.bias",
            # attn2 (cross-attention)
            "attn2.to_q.weight": "attn2.to_q.weight",
            "attn2.to_q.bias": "attn2.to_q.bias",
            "attn2.to_k.weight": "attn2.to_k.weight",
            "attn2.to_k.bias": "attn2.to_k.bias",
            "attn2.to_v.weight": "attn2.to_v.weight",
            "attn2.to_v.bias": "attn2.to_v.bias",
            "attn2.norm_q.weight": "attn2.norm_q.weight",
            "attn2.norm_k.weight": "attn2.norm_k.weight",
            "attn2.to_out.0.weight": "attn2.to_out.0.weight",
            "attn2.to_out.0.bias": "attn2.to_out.0.bias",
            # ff (GLUMBConv)
            "ff.conv_inverted.weight": "ff.conv_inverted.weight",
            "ff.conv_inverted.bias": "ff.conv_inverted.bias",
            "ff.conv_depth.weight": "ff.conv_depth.weight",
            "ff.conv_depth.bias": "ff.conv_depth.bias",
            "ff.conv_point.weight": "ff.conv_point.weight",
        }
        for hf_suffix, cactus_suffix in block_map.items():
            hf_key = f"{hf_prefix}.{hf_suffix}"
            if hf_key in sd:
                # scale_shift_table is used in add() (not matmul), so must stay FP16
                # to avoid INT8 N-padding causing shape mismatch [6,dim] -> [8,dim]
                w_precision = "FP16" if hf_suffix == "scale_shift_table" else precision
                save(sd[hf_key], out / f"{cactus_prefix}.{cactus_suffix}.weights", w_precision)
                if hf_suffix in {
                    "attn1.norm_q.weight",
                    "attn1.norm_k.weight",
                    "attn2.norm_q.weight",
                    "attn2.norm_k.weight",
                }:
                    qk_norm_exported += 1


        if (i + 1) % 7 == 0 or i == num_blocks - 1:
            print(f"  Converted block {i+1}/{num_blocks}")

    # ---- Final layer ----
    final_map = {
        "proj_out.weight": "final_layer.linear.weight",
        "proj_out.bias": "final_layer.linear.bias",
        "scale_shift_table": "final_layer.scale_shift_table",
    }
    for hf_key, cactus_key in final_map.items():
        if hf_key in sd:
            # scale_shift_table is used in add() (not matmul), must stay FP16
            w_precision = "FP16" if "scale_shift_table" in hf_key else precision
            save(sd[hf_key], out / f"{cactus_key}.weights", w_precision)
            print(f"  {hf_key} -> {cactus_key}")

    # Report unsaved weights
    saved_keys = set(global_map.keys()) | set(final_map.keys())
    for suffix in ["weight", "bias"]: saved_keys.add(f"patch_embed.proj.{suffix}")
    for i in range(num_blocks):
        p = f"transformer_blocks.{i}"
        for s in block_map:
            saved_keys.add(f"{p}.{s}")

    unsaved = set(sd.keys()) - saved_keys
    if unsaved:
        print(f"\n  WARNING: {len(unsaved)} unsaved transformer weights:")
        for k in sorted(unsaved)[:20]:
            print(f"    {k}: {sd[k].shape}")
        if len(unsaved) > 20:
            print(f"    ... and {len(unsaved) - 20} more")

    print(f"Transformer saved to {out}")
    transformer_config = transformer.config
    if getattr(transformer_config, "qk_norm", None) is not None and qk_norm_exported == 0:
        raise RuntimeError(
            "Transformer config enables qk_norm but no norm_q/norm_k weights were exported. "
            "Verify model-id/variant points to the intended Sana-Sprint checkpoint."
        )
    print(f"  qk_norm={getattr(transformer_config, 'qk_norm', None)} qk_norm_weights_exported={qk_norm_exported}")
    exported_config = {
        "hidden_dim": int(transformer_config.num_attention_heads * transformer_config.attention_head_dim),
        "num_layers": int(transformer_config.num_layers),
        "attention_heads": int(transformer_config.num_attention_heads),
        "attention_head_dim": int(transformer_config.attention_head_dim),
        "num_cross_attention_heads": int(transformer_config.num_cross_attention_heads),
        "cross_attention_head_dim": int(transformer_config.cross_attention_head_dim),
        "cross_attention_dim": int(transformer_config.cross_attention_dim),
        "caption_channels": int(transformer_config.caption_channels),
        "in_channels": int(transformer_config.in_channels),
        "patch_size": int(transformer_config.patch_size),
        "use_qk_norm": bool(getattr(transformer_config, "qk_norm", None)),
        "has_guidance_embeds": bool(getattr(transformer_config, "guidance_embeds", False)),
    }
    del transformer, sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return exported_config


# ──────────────────────────────────────────────
# 3. VAE (AutoencoderDC)
# ──────────────────────────────────────────────
def convert_vae(
    output_dir,
    precision="FP16",
    model_id="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    variant: Optional[str] = None,
):
    """Convert AutoencoderDC weights."""
    print(f"\n{'='*60}")
    print(f"Converting VAE (AutoencoderDC)")
    print(f"{'='*60}")

    from diffusers import AutoencoderDC
    vae, used_variant = load_diffusers_component(
        AutoencoderDC,
        model_id=model_id,
        subfolder="vae",
        variant=variant,
    )
    print(f"  Loaded VAE with variant={used_variant if used_variant else 'default'}")
    sd = vae.state_dict()

    vae_dir = Path(output_dir) / "vae"
    vae_encoder_dir = Path(output_dir) / "vae_encoder"
    vae_dir.mkdir(parents=True, exist_ok=True)
    vae_encoder_dir.mkdir(parents=True, exist_ok=True)

    # The HF decoder keys already use the exact naming convention Cactus expects:
    #   decoder.conv_in.weight -> conv_in.weight
    #   decoder.up_blocks.0.0.conv.weight -> up_blocks.0.0.conv.weight
    #   decoder.norm_out.weight -> norm_out.weight
    # We just strip the "decoder." prefix.
    decoder_count = 0
    encoder_count = 0
    for hf_key, tensor in sd.items():
        if hf_key.startswith("decoder."):
            cactus_key = hf_key.removeprefix("decoder.")
            save(tensor, vae_dir / f"{cactus_key}.weights", precision)
            decoder_count += 1
        elif hf_key.startswith("encoder."):
            # Keep native AutoencoderDC encoder names so Cactus can build optional true img2img latent encoding.
            cactus_key = hf_key.removeprefix("encoder.")
            save(tensor, vae_encoder_dir / f"{cactus_key}.weights", precision)
            encoder_count += 1

    print(f"  Saved {decoder_count} decoder weights and {encoder_count} encoder weights")

    print(f"VAE saved to {vae_dir}")
    del vae, sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def map_vae_decoder_key(hf_key):
    """Map HF AutoencoderDC decoder keys to Cactus convention.

    model_sana.cpp expects (with prefix="vae", folder=model_folder):
      {folder}/{prefix}.conv_in.weight.weights   -> project_in
      {folder}/{prefix}.conv_out.weight.weights   -> project_out ? or final conv
      {folder}/{prefix}.up_blocks.{i}.{j}.conv1.weight.weights
      {folder}/{prefix}.up_blocks.{i}.{j}.conv2.weight.weights
      {folder}/{prefix}.up_blocks.{i}.{j}.norm.weight.weights
      {folder}/{prefix}.up_blocks.{i}.0.conv.weight.weights  (DC up block)
      {folder}/{prefix}.norm_out.weight.weights
      {folder}/{prefix}.norm_out.bias.weights
    """
    k = hf_key.removeprefix("decoder.")

    # project_in -> conv_in
    if k.startswith("project_in."):
        return k.replace("project_in.", "conv_in.")

    # project_out -> conv_out
    if k.startswith("project_out.conv."):
        return k.replace("project_out.conv.", "conv_out.")

    # norm_out
    if k.startswith("project_out.norm."):
        return k.replace("project_out.norm.", "norm_out.")

    # stages.X.op_list.Y.* -> up_blocks.X.Y.*
    if k.startswith("stages."):
        parts = k.split(".")
        stage_idx = parts[1]
        # parts[2] should be "op_list"
        if parts[2] == "op_list":
            sub_idx = parts[3]
            rest = ".".join(parts[4:])
            return f"up_blocks.{stage_idx}.{sub_idx}.{rest}"

    print(f"  [decoder] unmapped key: {hf_key}")
    return None


def map_vae_encoder_key(hf_key):
    """Map HF AutoencoderDC encoder keys to Cactus convention.

    model_sana.cpp expects (with prefix="vae", folder=model_folder for encoder):
      {folder}/{prefix}.conv_in.weight.weights
      {folder}/{prefix}.down_blocks.{i}.{j}.conv1.weight.weights
      etc.
    """
    k = hf_key.removeprefix("encoder.")

    # project_in -> conv_in
    if k.startswith("project_in."):
        return "enc." + k.replace("project_in.", "conv_in.")

    # project_out -> conv_out
    if k.startswith("project_out."):
        return "enc." + k.replace("project_out.", "conv_out.")

    # stages.X.op_list.Y.* -> down_blocks.X.Y.*
    if k.startswith("stages."):
        parts = k.split(".")
        stage_idx = parts[1]
        if parts[2] == "op_list":
            sub_idx = parts[3]
            rest = ".".join(parts[4:])
            return f"enc.down_blocks.{stage_idx}.{sub_idx}.{rest}"

    print(f"  [encoder] unmapped key: {hf_key}")
    return None


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Convert Sana-Sprint weights to Cactus format")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--precision", default="FP16", choices=["FP16", "INT8", "INT4"],
                        help="Weight precision (default: FP16)")
    parser.add_argument("--model-id", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                        help="HuggingFace model ID for transformer + VAE")
    parser.add_argument(
        "--variant",
        default=None,
        choices=["fp16", "bf16"],
        help="Optional diffusers variant (e.g. fp16) for repos that do not expose default model files",
    )
    parser.add_argument("--text-encoder-id", default="google/gemma-2-2b-it",
                        help="HuggingFace model ID for text encoder")
    parser.add_argument("--skip-text-encoder", action="store_true",
                        help="Skip text encoder conversion")
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip transformer conversion")
    parser.add_argument("--skip-vae", action="store_true",
                        help="Skip VAE conversion")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each sub-model
    if not args.skip_text_encoder:
        convert_text_encoder(output_dir, args.precision, args.text_encoder_id)

    transformer_cfg = None
    if not args.skip_transformer:
        transformer_cfg = convert_transformer(output_dir, args.precision, args.model_id, args.variant)

    if not args.skip_vae:
        convert_vae(output_dir, args.precision, args.model_id, args.variant)

    # Write top-level config
    if transformer_cfg is None:
        # Fallback defaults for Sana 600M if transformer conversion was skipped.
        transformer_cfg = {
            "hidden_dim": 1152,
            "num_layers": 28,
            "attention_heads": 36,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 16,
            "cross_attention_head_dim": 72,
            "cross_attention_dim": 1152,
            "caption_channels": 2304,
            "in_channels": 32,
            "patch_size": 1,
            "use_qk_norm": False,
            "has_guidance_embeds": True,
        }

    config = {
        "model_type": "sana",
        "hidden_dim": transformer_cfg["hidden_dim"],
        "num_layers": transformer_cfg["num_layers"],
        "attention_heads": transformer_cfg["attention_heads"],
        "attention_head_dim": transformer_cfg["attention_head_dim"],
        "num_cross_attention_heads": transformer_cfg["num_cross_attention_heads"],
        "cross_attention_head_dim": transformer_cfg["cross_attention_head_dim"],
        "cross_attention_dim": transformer_cfg["cross_attention_dim"],
        "caption_channels": transformer_cfg["caption_channels"],
        "in_channels": transformer_cfg["in_channels"],
        "patch_size": transformer_cfg["patch_size"],
        "use_qk_norm": transformer_cfg.get("use_qk_norm", False),
        "has_guidance_embeds": transformer_cfg.get("has_guidance_embeds", True),
    }
    write_config(output_dir / "config.txt", config)
    print(f"\nConfig written to {output_dir / 'config.txt'}")

    print(f"\n{'='*60}")
    print(f"Conversion complete! Output: {output_dir}")
    print(f"{'='*60}")
    print(f"\nTo test:")
    print(f"  python -c \"from python.src.cactus import *; m = cactus_init('{output_dir}'); print(cactus_generate_image(m, 'a cat'))\"")


if __name__ == "__main__":
    main()
