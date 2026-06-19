# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view multi-modal MViT for driving maneuver anticipation."""

import math

import torch    
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .attention import get_rel_pos
from . import stem_helper
from .build import MODEL_REGISTRY
from .video_model_builder import MViT
from torch.utils.checkpoint import checkpoint as grad_ckpt


_MV_VIEWS = ("driver", "front", "left", "right", "rear")
_EV_VIEW = "ariagaze"
_INPUT_ORDER = _MV_VIEWS + (_EV_VIEW,)
_ALIASES = {
    "in_cabin": "driver",
    "incabin": "driver",
    "rearview": "rear",
    "ego": "ariagaze",
    "egoview": "ariagaze",
    "aria": "ariagaze",
    "gaze": "ariagaze",
    "aria_gaze": "ariagaze",
}


def _as_int(value):
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _attention_pool_with_special(tensor, pool, thw_shape, num_special=1, norm=None):
    if pool is None:
        return tensor, thw_shape

    tensor_dim = tensor.ndim
    if tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor_dim != 4:
        raise NotImplementedError("Unsupported input dimension {}".format(tensor.shape))

    special = tensor[:, :, :num_special, :] if num_special else None
    tensor = tensor[:, :, num_special:, :] if num_special else tensor

    b, n, _, c = tensor.shape
    t, h, w = thw_shape
    tensor = tensor.reshape(b * n, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    pooled_len = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(b, n, c, pooled_len).transpose(2, 3)
    if num_special:
        tensor = torch.cat((special, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


def _cal_rel_pos_spatial_with_special(
    attn, q, k, num_special, q_shape, k_shape, rel_pos_h, rel_pos_w
):
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)

    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h, device=q.device)[:, None] * q_h_ratio
        - torch.arange(k_h, device=q.device)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w, device=q.device)[:, None] * q_w_ratio
        - torch.arange(k_w, device=q.device)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)
    rh = rel_pos_h[dist_h.long()]
    rw = rel_pos_w[dist_w.long()]

    b, n_head, _, dim = q.shape
    grid_q = q[:, :, num_special:].reshape(b, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum("bythwc,hkc->bythwk", grid_q, rh)
    rel_w_q = torch.einsum("bythwc,wkc->bythwk", grid_q, rw)

    attn[:, :, num_special:, num_special:] = (
        attn[:, :, num_special:, num_special:].view(
            b, -1, q_t, q_h, q_w, k_t, k_h, k_w
        )
        + rel_h_q[:, :, :, :, :, None, :, None]
        + rel_w_q[:, :, :, :, :, None, None, :]
    ).view(b, -1, q_t * q_h * q_w, k_t * k_h * k_w)
    return attn


def _cal_rel_pos_temporal_with_special(
    attn, q, num_special, q_shape, k_shape, rel_pos_t
):
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = (
        torch.arange(q_t, device=q.device)[:, None] * q_t_ratio
        - torch.arange(k_t, device=q.device)[None, :] * k_t_ratio
    )
    dist_t += (k_t - 1) * k_t_ratio
    rt = rel_pos_t[dist_t.long()]

    b, n_head, _, dim = q.shape
    grid_q = q[:, :, num_special:].reshape(b, n_head, q_t, q_h, q_w, dim)
    grid_q = grid_q.permute(2, 0, 1, 3, 4, 5).reshape(
        q_t, b * n_head * q_h * q_w, dim
    )
    rel = torch.matmul(grid_q, rt.transpose(1, 2)).transpose(0, 1)
    rel = rel.view(b, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    attn[:, :, num_special:, num_special:] = (
        attn[:, :, num_special:, num_special:].view(
            b, -1, q_t, q_h, q_w, k_t, k_h, k_w
        )
        + rel[:, :, :, :, :, :, None, None]
    ).view(b, -1, q_t * q_h * q_w, k_t * k_h * k_w)
    return attn


def _attention_forward_with_special(attn_mod, x, thw_shape, num_special):
    b, n, _ = x.shape

    if attn_mod.pool_first:
        fold_dim = 1 if attn_mod.mode == "conv_unshared" else attn_mod.num_heads
        x = x.reshape(b, n, fold_dim, -1).permute(0, 2, 1, 3)
        q = k = v = x
    else:
        assert attn_mod.mode != "conv_unshared"
        if not attn_mod.separate_qkv:
            qkv = (
                attn_mod.qkv(x)
                .reshape(b, n, 3, attn_mod.num_heads, -1)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = attn_mod.q(x).reshape(b, n, attn_mod.num_heads, -1).permute(0, 2, 1, 3)
            k = attn_mod.k(x).reshape(b, n, attn_mod.num_heads, -1).permute(0, 2, 1, 3)
            v = attn_mod.v(x).reshape(b, n, attn_mod.num_heads, -1).permute(0, 2, 1, 3)

    q, q_shape = _attention_pool_with_special(
        q,
        attn_mod.pool_q,
        thw_shape,
        num_special=num_special,
        norm=getattr(attn_mod, "norm_q", None),
    )
    k, k_shape = _attention_pool_with_special(
        k,
        attn_mod.pool_k,
        thw_shape,
        num_special=num_special,
        norm=getattr(attn_mod, "norm_k", None),
    )
    v, v_shape = _attention_pool_with_special(
        v,
        attn_mod.pool_v,
        thw_shape,
        num_special=num_special,
        norm=getattr(attn_mod, "norm_v", None),
    )
    # print("q_shape =", q_shape)
    # print("k_shape =", k_shape)
    # print("q tensor =", q.shape)
    # print("k tensor =", k.shape)

    if attn_mod.pool_first:
        q_n = math.prod(q_shape) + num_special
        k_n = math.prod(k_shape) + num_special
        v_n = math.prod(v_shape) + num_special

        q = q.permute(0, 2, 1, 3).reshape(b, q_n, -1)
        q = attn_mod.q(q).reshape(b, q_n, attn_mod.num_heads, -1).permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3).reshape(b, k_n, -1)
        k = attn_mod.k(k).reshape(b, k_n, attn_mod.num_heads, -1).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3).reshape(b, v_n, -1)
        v = attn_mod.v(v).reshape(b, v_n, attn_mod.num_heads, -1).permute(0, 2, 1, 3)

    attn = (q * attn_mod.scale) @ k.transpose(-2, -1)
    if attn_mod.rel_pos_spatial:
        attn = _cal_rel_pos_spatial_with_special(
            attn,
            q,
            k,
            num_special,
            q_shape,
            k_shape,
            attn_mod.rel_pos_h,
            attn_mod.rel_pos_w,
        )
    if attn_mod.rel_pos_temporal:
        attn = _cal_rel_pos_temporal_with_special(
            attn,
            q,
            num_special,
            q_shape,
            k_shape,
            attn_mod.rel_pos_t,
        )

    attn = attn.softmax(dim=-1)
    x = attn @ v
    if attn_mod.residual_pooling:
        x[:, :, num_special:, :] += q[:, :, num_special:, :]
    x = x.transpose(1, 2).reshape(b, -1, attn_mod.dim_out)
    x = attn_mod.proj(x)
    if attn_mod.drop_rate > 0.0:
        x = attn_mod.proj_drop(x)
    return x, q_shape


def _block_forward_with_special(block, x, thw_shape, num_special):
    x_norm = block.norm1(x)
    x_block, thw_shape_new = _attention_forward_with_special(
        block.attn, x_norm, thw_shape, num_special
    )
    if block.dim_mul_in_att and block.dim != block.dim_out:
        x = block.proj(x_norm)
    x_res, _ = _attention_pool_with_special(
        x, block.pool_skip, thw_shape, num_special=num_special
    )
    if block.gamma_1 is not None:
        x = x_res + block.drop_path(block.gamma_1 * x_block)
    else:
        x = x_res + block.drop_path(x_block)

    x_norm = block.norm2(x)
    x_mlp = block.mlp(x_norm)
    if not block.dim_mul_in_att and block.dim != block.dim_out:
        x = block.proj(x_norm)
    if block.gamma_2 is not None:
        x = x + block.drop_path(block.gamma_2 * x_mlp)
    else:
        x = x + block.drop_path(x_mlp)
    return x, thw_shape_new


@MODEL_REGISTRY.register()
class M2MVT(nn.Module):
    """
    M2MVT with five multi-view streams and one ego-view/gaze stream.

    The MViTv2 encoder weights are shared by both branches. Stream-specific
    PatchEmbed projections produce branch tokens, learned memory tokens are
    prepended with the branch CLS token, and the whole sequence is encoded by
    the same MViT block stack before 7-class late fusion.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = MViT(cfg)
        import torch.nn as nn

        self.encoder.patch_embed = nn.Identity()
        self.encoder.head = nn.Identity()
        assert not self.encoder.enable_detection, "M2MVT supports classification only."
        assert not self.encoder.enable_rev, "M2MVT expects the non-reversible MViT path."

        self.input_order = _INPUT_ORDER
        in_channels = list(getattr(cfg.DATA, "INPUT_CHANNEL_NUM", [3]))
        if len(in_channels) < len(self.input_order):
            in_channels.extend([in_channels[-1]] * (len(self.input_order) - len(in_channels)))

        self.patch_embed = nn.ModuleDict(
            {
                name: stem_helper.PatchEmbed(
                    dim_in=in_channels[idx],
                    dim_out=cfg.MVIT.EMBED_DIM,
                    kernel=cfg.MVIT.PATCH_KERNEL,
                    stride=cfg.MVIT.PATCH_STRIDE,
                    padding=cfg.MVIT.PATCH_PADDING,
                    conv_2d=cfg.MVIT.PATCH_2D,
                )
                for idx, name in enumerate(self.input_order)
            }
        )

        embed_dim = cfg.MVIT.EMBED_DIM
        dim = self.encoder.norm.normalized_shape[0]
        dropout_rate = float(getattr(cfg.MODEL, "DROPOUT_RATE", 0.0))
        self.mv_memory = nn.Parameter(torch.zeros(1, 10, embed_dim))
        self.ev_memory = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.head_drop = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.head = nn.Linear(2 * dim, 7)
        trunc_normal_(self.mv_memory, std=0.02)
        trunc_normal_(self.ev_memory, std=0.02)
        trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = ["mv_memory", "ev_memory"]
        if hasattr(self.encoder, "no_weight_decay"):
            names.extend(["encoder.{}".format(name) for name in self.encoder.no_weight_decay()])
        return names

    def _canonical_inputs(self, x):
        if isinstance(x, dict):
            canonical = {}
            for key, value in x.items():
                name = _ALIASES.get(key, key)
                canonical[name] = value
            missing = [name for name in self.input_order if name not in canonical]
            if missing:
                raise KeyError("Missing M2MVT input view(s): {}".format(", ".join(missing)))
            return {name: canonical[name] for name in self.input_order}

        if isinstance(x, (list, tuple)):
            if len(x) == 1 and isinstance(x[0], dict):
                return self._canonical_inputs(x[0])
            if len(x) != len(self.input_order):
                raise ValueError(
                    "M2MVT expects {} inputs in order {}; got {}.".format(
                        len(self.input_order), self.input_order, len(x)
                    )
                )
            return {name: x[idx] for idx, name in enumerate(self.input_order)}

        raise TypeError("M2MVT input must be a dict or a 6-item list/tuple.")

    def _patchify(self, name, video):
        tokens, bcthw = self.patch_embed[name](video)
        bcthw = list(bcthw)
        if len(bcthw) == 4:
            bcthw.insert(2, torch.tensor(self.encoder.T, device=video.device))

        t, h, w = [_as_int(v) for v in bcthw[-3:]]
        expected = (self.encoder.T, self.encoder.H, self.encoder.W)
        if (t, h, w) != expected:
            raise ValueError(
                "M2MVT expects patch geometry {}; {} produced {}.".format(
                    expected, name, (t, h, w)
                )
            )
        # print(name, tokens.shape, bcthw)    
        return tokens, bcthw, [t, h, w]

    def _full_pos_embed(self):
        if self.encoder.sep_pos_embed:
            pos_embed = self.encoder.pos_embed_spatial.repeat(
                1, self.encoder.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.encoder.pos_embed_temporal,
                self.encoder.patch_dims[1] * self.encoder.patch_dims[2],
                dim=1,
            )
            if self.encoder.cls_embed_on:
                pos_embed = torch.cat([self.encoder.pos_embed_class, pos_embed], dim=1)
            return pos_embed
        return self.encoder.pos_embed

    def _add_patch_pos_embed(self, tokens, bcthw):
        if self.encoder.use_fixed_sincos_pos:
            start = 1 if self.encoder.cls_embed_on else 0
            tokens = tokens + self.encoder.pos_embed[:, start:, :]

        if self.encoder.use_abs_pos:
            pos_embed = self.encoder._get_pos_embed(self._full_pos_embed(), bcthw)
            if self.encoder.cls_embed_on:
                pos_embed = pos_embed[:, 1:, :]
            tokens = tokens + pos_embed
        return tokens

    def _cls_token(self, batch_size, bcthw):
        if not self.encoder.cls_embed_on:
            return None

        cls_token = self.encoder.cls_token.expand(batch_size, -1, -1)
        if self.encoder.use_fixed_sincos_pos:
            cls_token = cls_token + self.encoder.pos_embed[:, :1, :]
        if self.encoder.use_abs_pos:
            pos_embed = self.encoder._get_pos_embed(self._full_pos_embed(), bcthw)
            cls_token = cls_token + pos_embed[:, :1, :]
        return cls_token

    def _encode_token_sequence(self, patch_tokens, bcthw, thw, memory_tokens):
        
        batch_size = patch_tokens.shape[0]
        cls_token = self._cls_token(batch_size, bcthw)
        memory_tokens = memory_tokens.expand(batch_size, -1, -1)

        if cls_token is not None:
            patch_tokens = torch.cat((cls_token, memory_tokens, patch_tokens), dim=1)
            num_special = 1 + memory_tokens.shape[1]
        else:
            patch_tokens = torch.cat((memory_tokens, patch_tokens), dim=1)
            num_special = memory_tokens.shape[1]

        if self.encoder.drop_rate:
            patch_tokens = self.encoder.pos_drop(patch_tokens)
        if self.encoder.norm_stem:
            patch_tokens = self.encoder.norm_stem(patch_tokens)

        for i, block in enumerate(self.encoder.blocks):
            patch_tokens, thw = grad_ckpt(
                _block_forward_with_special,
                block, patch_tokens, thw, num_special,
                use_reentrant=False,
            )

        if self.encoder.use_mean_pooling:
            patch_tokens = patch_tokens[:, num_special:]
            cls_embedding = self.encoder.norm(patch_tokens.mean(1))
        elif self.encoder.cls_embed_on:
            cls_embedding = self.encoder.norm(patch_tokens)[:, 0]
        else:
            cls_embedding = self.encoder.norm(patch_tokens[:, num_special:]).mean(1)
        return cls_embedding

    def _encode_views(self, inputs, names, memory_tokens):
        all_tokens = []
        base_bcthw = None
        base_thw = None
        for name in names:
            tokens, bcthw, thw = self._patchify(name, inputs[name])
            tokens = self._add_patch_pos_embed(tokens, bcthw)
            all_tokens.append(tokens)
            if base_bcthw is None:
                base_bcthw = bcthw
                base_thw = thw

        tokens = torch.cat(all_tokens, dim=1)
        # print("token shape =", tokens.shape)
        branch_thw = [base_thw[0] * len(names), base_thw[1], base_thw[2]]
        return self._encode_token_sequence(tokens, base_bcthw, branch_thw, memory_tokens)

    def forward(self, x, bboxes=None):
        del bboxes
        inputs = self._canonical_inputs(x)

        mv_cls = self._encode_views(inputs, _MV_VIEWS, self.mv_memory)
        ev_cls = self._encode_views(inputs, (_EV_VIEW,), self.ev_memory)

        fused = torch.cat((mv_cls, ev_cls), dim=1)
        return self.head(self.head_drop(fused))
