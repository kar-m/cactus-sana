#!/usr/bin/env python3
"""
Convert Sana AutoencoderDC decoder to CoreML (.mlpackage) for Apple ANE acceleration.

Usage:
    python python/convert_sana_coreml.py --output ./weights/sana-0.6b
    python python/convert_sana_coreml.py --output ./weights/sana-0.6b --validate --latent-size 32

Prerequisites:
    pip install torch diffusers coremltools
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Trace-friendly replacements for unsupported ops
# ──────────────────────────────────────────────

def _movedim_to_permute(x: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    """Replace movedim (unsupported in CoreML) with permute."""
    ndim = x.ndim
    if src < 0:
        src += ndim
    if dst < 0:
        dst += ndim
    perm = list(range(ndim))
    perm.pop(src)
    perm.insert(dst, src)
    return x.permute(perm)

def _repeat_interleave_trace_friendly(x: torch.Tensor, repeats: int, dim: int) -> torch.Tensor:
    """Replace torch.repeat_interleave with repeat+reshape that CoreML handles."""
    # For dim=1 (channel): [N, C, H, W] -> [N, C, r, H, W] -> [N, C*r, H, W]
    # Use torch.repeat_interleave which coremltools supports for scalar repeats
    return torch.repeat_interleave(x, repeats, dim=dim)


def _interpolate_nearest_2x(x: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor 2x upsample. F.interpolate with scale_factor traces cleanly to CoreML."""
    return F.interpolate(x, scale_factor=2, mode="nearest")


class TraceFriendlyRMSNorm(nn.Module):
    """RMSNorm that stays in FP16 (no float32 upcast) for ANE compatibility."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.weight = original.weight
        self.bias = getattr(original, "bias", None)
        self.eps = original.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class TraceFriendlyLinearAttention(nn.Module):
    """Linear attention rewritten to avoid float32 upcast inside the trace."""

    def __init__(self, original):
        super().__init__()
        self.eps = original.eps

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Original pads value with 1s for normalization trick
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        scores = torch.matmul(value, key.transpose(-1, -2))
        hidden = torch.matmul(scores, query)
        # Normalize: all channels / last channel (the padding-1 row sums)
        return hidden[:, :, :-1] / (hidden[:, :, -1:] + self.eps)


class TraceFriendlyDCUpBlock(nn.Module):
    """DCUpBlock2d with repeat_interleave and F.interpolate replaced."""

    def __init__(self, original):
        super().__init__()
        self.conv = original.conv
        self.interpolate = original.interpolate
        self.factor = original.factor
        self.shortcut = original.shortcut
        self.repeats = original.repeats

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = _interpolate_nearest_2x(hidden_states)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = _repeat_interleave_trace_friendly(hidden_states, self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x
        return hidden_states


class TraceFriendlyAttnProcessor:
    """Attention processor that avoids dynamic shapes and float32 upcast for CoreML."""

    def __init__(self, linear_attn_module):
        self.linear_attn = linear_attn_module

    def __call__(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use permute instead of movedim, avoid dynamic shape ops
        residual = hidden_states

        # [N, C, H, W] -> [N, H, W, C]
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        # [N, H, W, 3*C] -> [N, 3*C, H, W]
        hidden_states = torch.cat([query, key, value], dim=3).permute(0, 3, 1, 2)

        multi_scale_qkv = [hidden_states]
        for block in attn.to_qkv_multiscale:
            multi_scale_qkv.append(block(hidden_states))
        hidden_states = torch.cat(multi_scale_qkv, dim=1)

        # [N, 6*C, H, W] -> [N, 6*C, H*W] -> [N, 2*heads, 3*head_dim, H*W]
        H, W = hidden_states.shape[2], hidden_states.shape[3]
        hidden_states = hidden_states.flatten(2)  # [N, 6*C, H*W]
        hidden_states = hidden_states.unflatten(1, (-1, 3 * attn.attention_head_dim))  # [N, 2*heads, 3*head_dim, H*W]
        query, key, value = hidden_states.chunk(3, dim=2)
        query = attn.nonlinearity(query)
        key = attn.nonlinearity(key)

        hidden_states = self.linear_attn(query, key, value)

        # [N, 2*heads, head_dim, H*W] -> [N, 2*C, H*W] -> [N, 2*C, H, W]
        hidden_states = hidden_states.flatten(1, 2)  # [N, 2*C, H*W]
        hidden_states = hidden_states.unflatten(2, (H, W))  # [N, 2*C, H, W]
        # [N, 2*C, H, W] -> [N, H, W, 2*C] -> to_out -> [N, H, W, C] -> [N, C, H, W]
        hidden_states = attn.to_out(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if attn.norm_type == "rms_norm":
            hidden_states = attn.norm_out(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            hidden_states = attn.norm_out(hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states


# ──────────────────────────────────────────────
# Model patching
# ──────────────────────────────────────────────

def patch_decoder_for_trace(decoder: nn.Module) -> nn.Module:
    """Recursively replace unsupported modules/ops with trace-friendly versions."""

    # 1. Replace all RMSNorm modules
    _replace_modules_by_class(decoder, "RMSNorm", lambda m: TraceFriendlyRMSNorm(m))

    # 2. Replace DCUpBlock2d modules
    _replace_modules_by_class(decoder, "DCUpBlock2d", lambda m: TraceFriendlyDCUpBlock(m))

    # 3. Replace attention processors and linear attention
    for name, module in decoder.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == "SanaMultiscaleLinearAttention":
            trace_linear_attn = TraceFriendlyLinearAttention(module)
            module.apply_linear_attention = trace_linear_attn.forward
            module.processor = TraceFriendlyAttnProcessor(trace_linear_attn)

    # 4. Patch Decoder.forward to remove output_size from repeat_interleave
    #    (output_size causes dynamic int cast that CoreML can't handle)
    if hasattr(decoder, "in_shortcut") and decoder.in_shortcut:
        original_forward = decoder.forward
        repeats = decoder.in_shortcut_repeats

        def patched_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            x = torch.repeat_interleave(hidden_states, repeats, dim=1)
            hidden_states = decoder.conv_in(hidden_states) + x

            for up_block in reversed(decoder.up_blocks):
                hidden_states = up_block(hidden_states)

            hidden_states = decoder.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
            hidden_states = decoder.conv_act(hidden_states)
            hidden_states = decoder.conv_out(hidden_states)
            return hidden_states

        decoder.forward = patched_forward

    return decoder


def _replace_modules_by_class(root: nn.Module, class_name: str, factory):
    """Replace all submodules matching class_name using factory(original) -> replacement."""
    for parent_name, parent in list(root.named_modules()):
        for attr_name, child in list(parent.named_children()):
            if child.__class__.__name__ == class_name:
                replacement = factory(child)
                setattr(parent, attr_name, replacement)
        # Handle Sequential/ModuleList items
        if isinstance(parent, (nn.Sequential, nn.ModuleList)):
            for i, child in enumerate(parent):
                if child.__class__.__name__ == class_name:
                    parent[i] = factory(child)


# ──────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────

def convert_vae_decoder_to_coreml(
    output_dir: str,
    model_id: str = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    latent_size: int = 32,
    latent_channels: int = 32,
    validate: bool = False,
):
    """Convert AutoencoderDC decoder to CoreML .mlpackage."""
    import coremltools as ct
    from diffusers import AutoencoderDC

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlpackage_path = output_dir / "vae_decoder.mlpackage"

    print(f"\n{'='*60}")
    print("Converting VAE decoder to CoreML")
    print(f"{'='*60}")
    print(f"  Model: {model_id}")
    print(f"  Latent shape: [1, {latent_channels}, {latent_size}, {latent_size}]")

    # Load VAE
    print("  Loading AutoencoderDC...")
    # Load in float32 for fast CPU tracing, coremltools handles FP16 conversion
    vae = AutoencoderDC.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    decoder = vae.decoder.eval().float()

    # Keep a reference copy for validation before patching
    if validate:
        import copy
        decoder_ref = copy.deepcopy(vae.decoder).eval().float()

    # Patch for trace compatibility
    print("  Patching decoder for CoreML trace...")
    decoder = patch_decoder_for_trace(decoder)

    # Trace in float32 (much faster on CPU than float16)
    dummy_input = torch.randn(1, latent_channels, latent_size, latent_size, dtype=torch.float32)
    print(f"  Tracing with input shape {list(dummy_input.shape)}...")
    t0 = time.time()

    # Monkey-patch movedim -> permute (movedim not supported in coremltools)
    _original_movedim = torch.Tensor.movedim
    torch.Tensor.movedim = lambda self, src, dst: _movedim_to_permute(self, src, dst)

    with torch.no_grad():
        traced = torch.jit.trace(decoder, dummy_input)

    torch.Tensor.movedim = _original_movedim

    print(f"  Traced in {time.time() - t0:.1f}s")

    # Validate trace output matches original
    if validate:
        print("  Validating trace accuracy...")
        with torch.no_grad():
            ref_out = decoder_ref(dummy_input)
            traced_out = traced(dummy_input)
        max_err = (ref_out.float() - traced_out.float()).abs().max().item()
        print(f"  Max error (original vs traced): {max_err:.6f}")
        if max_err > 0.05:
            print(f"  WARNING: high error, traced model may be inaccurate")
        del decoder_ref

    # Convert to CoreML
    print("  Converting to CoreML mlprogram...")
    t0 = time.time()

    latent_shape = (1, latent_channels, latent_size, latent_size)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latents", shape=latent_shape)],
        outputs=[ct.TensorType(name="rgb")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )
    print(f"  Converted in {time.time() - t0:.1f}s")

    # Save
    mlmodel.save(str(mlpackage_path))
    print(f"  Saved to {mlpackage_path}")

    # Validate CoreML output
    if validate:
        print("  Validating CoreML prediction...")
        try:
            pred = mlmodel.predict({"latents": dummy_input.numpy()})
            coreml_out = pred["rgb"]
            traced_np = traced(dummy_input).detach().numpy()
            max_err = np.abs(coreml_out.astype(np.float32) - traced_np.astype(np.float32)).max()
            print(f"  Max error (traced vs CoreML): {max_err:.6f}")
        except Exception as e:
            print(f"  CoreML validation skipped: {e}")

    # Cleanup
    del vae, decoder, traced, mlmodel
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'='*60}")
    print(f"CoreML conversion complete: {mlpackage_path}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Sana VAE decoder to CoreML")
    parser.add_argument("--output", "-o", required=True, help="Output directory (e.g., ./weights/sana-0.6b)")
    parser.add_argument("--model-id", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                        help="HuggingFace model ID")
    parser.add_argument("--latent-size", type=int, default=32,
                        help="Latent spatial size (32 for 1024px, 16 for 512px)")
    parser.add_argument("--latent-channels", type=int, default=32,
                        help="Number of latent channels")
    parser.add_argument("--validate", action="store_true",
                        help="Validate accuracy after conversion")
    args = parser.parse_args()

    convert_vae_decoder_to_coreml(
        output_dir=args.output,
        model_id=args.model_id,
        latent_size=args.latent_size,
        latent_channels=args.latent_channels,
        validate=args.validate,
    )


if __name__ == "__main__":
    main()
