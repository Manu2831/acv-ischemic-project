"""
ViT 3D (desde cero) o tokens DINOv2 (preentrenado, congelado) + codificador tabular + fusiones.

Imagen (B, C, D, H, W) -> tokenizador -> grid de tokens (D/4, H/8, W/8) con
posicional 3D sin-cos fijo (cada token sabe su posición z, y, x en el cerebro)
-> bloques transformer -> attention pooling (deja un mapa espacial de "dónde mira").

Tabular (B, F): cada variable clínica es un token (estilo FT-Transformer), así
puede participar en cross-attention con los tokens de imagen.

Fusiones (cuando hay imagen + tabular):
  concat     : alinear (misma dim + LayerNorm) y concatenar vectores -> cabeza.
  film       : el tabular modula los tokens de imagen (escala/desplazamiento por
               canal) ANTES del transformer, estilo DAFT (Pölsterl et al. 2021).
  cross_attn : cada token clínico consulta a los tokens de imagen ("¿dónde en el
               cerebro está lo relevante para este NIHSS/edad?").
  gated      : una cabeza por modalidad + red de compuertas que decide por
               paciente cuánto confiar en cada una; pérdidas auxiliares evitan
               que una modalidad se ignore (lo que pasó con BERT).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = (4, 8, 8)  # reducción total (D, H, W) del tokenizador


def sincos_pos_embed_3d(dim, grid):
    d_axis = (dim // 6) * 2

    def axis(n):
        pos = torch.arange(n, dtype=torch.float32)
        omega = 1.0 / (10000 ** (torch.arange(d_axis // 2, dtype=torch.float32) / (d_axis // 2)))
        out = pos[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    D, H, W = grid
    ez, ey, ex = axis(D), axis(H), axis(W)
    emb = torch.cat([
        ez[:, None, None].expand(D, H, W, d_axis),
        ey[None, :, None].expand(D, H, W, d_axis),
        ex[None, None, :].expand(D, H, W, d_axis),
    ], dim=-1).reshape(D * H * W, 3 * d_axis)
    return F.pad(emb, (0, dim - emb.shape[1]))


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) >= self.p).to(x.dtype)
        return x * keep / (1 - self.p)


class Attention(nn.Module):
    def __init__(self, dim, heads, attn_drop=0.0):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

    def forward(self, x, context=None, return_attn=False):
        ctx = x if context is None else context
        B, N, C = x.shape
        h = self.heads
        q = self.q(x).view(B, N, h, C // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, h, C // h).permute(2, 0, 3, 1, 4)
        attn = None
        if return_attn:
            attn = ((q @ k.transpose(-2, -1)) / math.sqrt(C // h)).softmax(-1)
            out = F.dropout(attn, self.attn_drop, self.training) @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, N, C)), attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, drop=0.1, drop_path=0.0, cross=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim) if cross else None
        self.attn = Attention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, dim), nn.Dropout(drop))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, context=None, return_attn=False):
        ctx = self.norm_ctx(context) if context is not None else None
        a, attn = self.attn(self.norm1(x), ctx, return_attn)
        x = x + self.drop_path(a)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.norm(x)
        w = self.score(x).squeeze(-1).softmax(-1)
        return (w.unsqueeze(-1) * x).sum(1), w


def conv_gn_act(cin, cout, stride):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(min(8, cout), cout), nn.GELU())


class ImageEncoder(nn.Module):
    """tokenizer='conv': stem convolucional (sesgo de localidad, mejor con pocos datos, tipo CCT).
    tokenizer='patch': ViT puro, parches no solapados de PATCH vóxeles."""

    def __init__(self, in_ch, input_shape, dim=96, depth=4, heads=4, mlp_ratio=2.0,
                 drop=0.1, drop_path=0.1, tokenizer="conv"):
        super().__init__()
        if any(s % p for s, p in zip(input_shape, PATCH)):
            raise ValueError(f"input_shape {input_shape} debe ser múltiplo de {PATCH}")
        if tokenizer == "conv":
            self.tokenizer = nn.Sequential(conv_gn_act(in_ch, dim // 4, (1, 2, 2)),
                                           conv_gn_act(dim // 4, dim // 2, (2, 2, 2)),
                                           conv_gn_act(dim // 2, dim, (2, 2, 2)))
        else:
            self.tokenizer = nn.Conv3d(in_ch, dim, kernel_size=PATCH, stride=PATCH)
        self.grid = tuple(s // p for s, p in zip(input_shape, PATCH))
        self.register_buffer("pos", sincos_pos_embed_3d(dim, self.grid), persistent=False)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, drop, dpr[i]) for i in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.pool = AttentionPool(dim)

    def forward(self, x, film=None):
        t = self.tokenizer(x).flatten(2).transpose(1, 2) + self.pos
        if film is not None:
            gamma, beta = film
            t = t * (1 + gamma[:, None]) + beta[:, None]
        for blk in self.blocks:
            t, _ = blk(t)
        return self.norm(t)


class FeatureTokenEncoder(nn.Module):
    """Tokens precalculados por un ViT 2D congelado (DINOv2), corte por corte.
    Entrada (B, T, feat_dim + n_aux): features de cada token + fracción de lesión del token.
    Agrega posición 3D (corte, fila, columna) y un transformer chico entre cortes."""

    def __init__(self, feat_dim, n_aux, grid, dim=96, depth=2, heads=4, mlp_ratio=2.0, drop=0.1, drop_path=0.1):
        super().__init__()
        self.feat_dim, self.grid = feat_dim, tuple(grid)
        self.in_norm = nn.LayerNorm(feat_dim)
        self.proj = nn.Sequential(nn.Linear(feat_dim, dim), nn.Dropout(drop))
        self.aux_proj = nn.Linear(n_aux, dim) if n_aux else None
        self.register_buffer("pos", sincos_pos_embed_3d(dim, self.grid), persistent=False)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, drop, dpr[i]) for i in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.pool = AttentionPool(dim)

    def forward(self, x, film=None):
        t = self.proj(self.in_norm(x[..., :self.feat_dim]))
        if self.aux_proj is not None:
            t = t + self.aux_proj(x[..., self.feat_dim:])
        t = t + self.pos
        if film is not None:
            gamma, beta = film
            t = t * (1 + gamma[:, None]) + beta[:, None]
        for blk in self.blocks:
            t, _ = blk(t)
        return self.norm(t)


class TabularEncoder(nn.Module):
    def __init__(self, n_features, dim=96, depth=1, heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        bound = 1 / math.sqrt(dim)
        self.weight = nn.Parameter(torch.empty(n_features, dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_features, dim).uniform_(-bound, bound))
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, drop) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        t = x.unsqueeze(-1) * self.weight + self.bias
        for blk in self.blocks:
            t, _ = blk(t)
        t = self.norm(t)
        return t, t.mean(1)


def make_head(in_dim, hidden, drop):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(drop), nn.Linear(in_dim, hidden),
                         nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))


class MultimodalNet(nn.Module):
    FUSIONS = ("concat", "film", "cross_attn", "gated")

    def __init__(self, in_ch, input_shape, n_tab, fusion="concat", dim=96, depth=4, heads=4,
                 tab_depth=1, cross_depth=1, mlp_ratio=2.0, drop=0.1, drop_path=0.1,
                 head_drop=0.3, tokenizer="conv", encoder="vit3d", feat_dim=0, n_aux=0, token_grid=None):
        """encoder='vit3d': in_ch canales de volumen (B, C, D, H, W), entrenado desde cero.
        encoder='dinov2'/'lesion': tokens precalculados (B, T, feat_dim + n_aux); in_ch = nº de modalidades."""
        super().__init__()
        self.use_img, self.use_tab = in_ch > 0, n_tab > 0
        if not (self.use_img or self.use_tab):
            raise ValueError("Se necesita al menos imagen o tabular")
        self.fusion = fusion if (self.use_img and self.use_tab) else "none"
        if self.fusion not in self.FUSIONS + ("none",):
            raise ValueError(f"fusion desconocida: {fusion}")

        if self.use_img and encoder != "vit3d":
            self.img = FeatureTokenEncoder(feat_dim, n_aux, token_grid, dim, depth, heads, mlp_ratio, drop, drop_path)
        elif self.use_img:
            self.img = ImageEncoder(in_ch, input_shape, dim, depth, heads, mlp_ratio, drop, drop_path, tokenizer)
        if self.use_tab:
            self.tab = TabularEncoder(n_tab, dim, tab_depth, heads, mlp_ratio, drop)

        if self.fusion == "film":
            self.film = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2 * dim))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        if self.fusion == "cross_attn":
            self.cross = nn.ModuleList(Block(dim, heads, mlp_ratio, drop, cross=True) for _ in range(cross_depth))
            self.cross_norm = nn.LayerNorm(dim)

        if self.fusion == "gated":
            self.head_img = make_head(dim, dim // 2, head_drop)
            self.head_tab = make_head(dim, dim // 2, head_drop)
            self.gate = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim // 2), nn.GELU(),
                                      nn.Dropout(head_drop), nn.Linear(dim // 2, 2))
        else:
            in_dim = dim if self.fusion == "none" else 2 * dim
            self.head = make_head(in_dim, dim // 2, head_drop)

    def forward(self, img=None, tab=None):
        out = {}
        if self.use_tab:
            tab_tokens, tab_vec = self.tab(tab)
        if self.use_img:
            film = self.film(tab_vec).chunk(2, dim=-1) if self.fusion == "film" else None
            img_tokens = self.img(img, film)
            img_vec, out["img_attn"] = self.img.pool(img_tokens)

        if self.fusion == "none":
            z = img_vec if self.use_img else tab_vec
            out["logit"] = self.head(z)
        elif self.fusion in ("concat", "film"):
            z = torch.cat([img_vec, tab_vec], dim=-1)
            out["logit"] = self.head(z)
        elif self.fusion == "cross_attn":
            q = tab_tokens
            for blk in self.cross:
                q, attn = blk(q, context=img_tokens, return_attn=True)
            out["cross_attn"] = attn.mean(1)  # (B, F, N_tokens): a qué región mira cada variable clínica
            z = torch.cat([img_vec, self.cross_norm(q).mean(1)], dim=-1)
            out["logit"] = self.head(z)
        else:  # gated
            li, lt = self.head_img(img_vec), self.head_tab(tab_vec)
            z = torch.cat([img_vec, tab_vec], dim=-1)
            g = self.gate(z).softmax(-1)
            out["logit"] = g[:, :1] * li + g[:, 1:] * lt
            out["aux_logits"] = (li, lt)
            out["gates"] = g
        out["emb"] = z
        return out
