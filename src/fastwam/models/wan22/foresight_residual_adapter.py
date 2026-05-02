from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .wan_video_dit import sinusoidal_embedding_1d


def _validate_adapter_inputs(base_action: torch.Tensor, timestep: torch.Tensor):
    if base_action.ndim != 3:
        raise ValueError(
            f"`base_action` must be [B, T, A], got shape {tuple(base_action.shape)}"
        )
    if timestep.ndim != 1:
        raise ValueError(f"`timestep` must be [B], got shape {tuple(timestep.shape)}")
    if timestep.shape[0] != base_action.shape[0]:
        raise ValueError(
            f"`timestep` batch must match `base_action`: {timestep.shape[0]} vs {base_action.shape[0]}"
        )


class MLPResidualAdapter(nn.Module):
    """Output-side token-wise residual head used by the current PFD-S1 baseline."""

    def __init__(
        self,
        action_dim: int,
        freq_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError(f"`depth` must be >= 2, got {depth}")

        self.action_dim = int(action_dim)
        self.freq_dim = int(freq_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        self.action_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        layers: list[nn.Module] = [
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(self.depth - 2):
            layers.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Linear(self.hidden_dim, self.action_dim))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, base_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        _validate_adapter_inputs(base_action, timestep)
        action_feat = self.action_proj(base_action)
        time_feat = self.time_proj(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(
                device=base_action.device,
                dtype=base_action.dtype,
            )
        )
        time_feat = time_feat.unsqueeze(1).expand(-1, base_action.shape[1], -1)
        return self.mlp(torch.cat([action_feat, time_feat], dim=-1))


class ForesightResidualAdapter(MLPResidualAdapter):
    """Backward-compatible alias for the current output-only MLP adapter."""


class FeatureResidualAdapter(nn.Module):
    """Residual MLP adapter for action hidden states in later action-expert layers."""

    def __init__(
        self,
        hidden_dim: int,
        freq_dim: int,
        depth: int = 2,
        mlp_mult: int = 2,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError(f"`depth` must be >= 2, got {depth}")
        if mlp_mult < 1:
            raise ValueError(f"`mlp_mult` must be >= 1, got {mlp_mult}")

        self.hidden_dim = int(hidden_dim)
        self.freq_dim = int(freq_dim)
        self.depth = int(depth)
        self.mlp_mult = int(mlp_mult)
        inner_dim = self.hidden_dim * self.mlp_mult

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 3),
        )
        nn.init.zeros_(self.time_proj[-1].weight)
        nn.init.zeros_(self.time_proj[-1].bias)

        layers: list[nn.Module] = [
            nn.Linear(self.hidden_dim, inner_dim),
            nn.GELU(),
        ]
        for _ in range(self.depth - 2):
            layers.extend(
                [
                    nn.Linear(inner_dim, inner_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(inner_dim, self.hidden_dim))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"`hidden_states` must be [B, T, D], got shape {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"`hidden_states` last dim must be {self.hidden_dim}, got {hidden_states.shape[-1]}"
            )
        if timestep.ndim != 1 or timestep.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                f"`timestep` must be [B] and match hidden-state batch, got shape {tuple(timestep.shape)}"
            )

        time_feat = sinusoidal_embedding_1d(self.freq_dim, timestep).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        shift, scale, gate = self.time_proj(time_feat).chunk(3, dim=-1)
        adapted = self.mlp(_modulate(self.norm(hidden_states), shift, scale))
        return hidden_states + gate.unsqueeze(1) * adapted


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TemporalSpatialResidualBlock(nn.Module):
    """Temporal attention + local temporal mixer + channel FFN with timestep conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_mult: int,
        dropout: float,
        local_kernel_size: int,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` must be divisible by `num_heads`, got {hidden_dim} and {num_heads}"
            )
        if local_kernel_size < 1 or local_kernel_size % 2 == 0:
            raise ValueError(
                f"`local_kernel_size` must be a positive odd integer, got {local_kernel_size}"
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(hidden_dim * ffn_mult)

        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_local = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)

        self.local_depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=local_kernel_size,
            padding=local_kernel_size // 2,
            groups=hidden_dim,
        )
        self.local_pointwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, hidden_dim),
        )

        self.time_to_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 9),
        )
        nn.init.zeros_(self.time_to_mod[-1].weight)
        nn.init.zeros_(self.time_to_mod[-1].bias)

    def forward(self, x: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        (
            attn_shift,
            attn_scale,
            attn_gate,
            local_shift,
            local_scale,
            local_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = self.time_to_mod(time_feat).chunk(9, dim=-1)

        attn_input = _modulate(self.norm_attn(x), attn_shift, attn_scale)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_gate.unsqueeze(1) * self.attn_out(attn_out)

        local_input = _modulate(self.norm_local(x), local_shift, local_scale)
        local_input = local_input.transpose(1, 2)
        local_out = self.local_depthwise(local_input)
        local_out = self.local_pointwise(local_out).transpose(1, 2)
        x = x + local_gate.unsqueeze(1) * local_out

        ffn_input = _modulate(self.norm_ffn(x), ffn_shift, ffn_scale)
        x = x + ffn_gate.unsqueeze(1) * self.ffn(ffn_input)
        return x


class TemporalSpatialResidualAdapter(nn.Module):
    """Output-side residual head with temporal attention and local horizon mixing."""

    def __init__(
        self,
        action_dim: int,
        freq_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        max_horizon: int = 128,
        local_kernel_size: int = 3,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"`num_layers` must be >= 1, got {num_layers}")

        self.action_dim = int(action_dim)
        self.freq_dim = int(freq_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_horizon = int(max_horizon)

        self.input_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_horizon, self.hidden_dim))
        nn.init.normal_(self.temporal_pos, mean=0.0, std=0.02)

        self.time_proj = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                TemporalSpatialResidualBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    local_kernel_size=local_kernel_size,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, base_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        _validate_adapter_inputs(base_action, timestep)
        batch, horizon, _ = base_action.shape
        if horizon > self.max_horizon:
            raise ValueError(
                f"`base_action` horizon {horizon} exceeds adapter max_horizon {self.max_horizon}"
            )

        time_feat = self.time_proj(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(
                device=base_action.device,
                dtype=base_action.dtype,
            )
        )
        x = self.input_proj(base_action)
        x = x + self.temporal_pos[:, :horizon].to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, time_feat)
        return self.output_proj(self.final_norm(x))


def build_foresight_residual_adapter(
    *,
    action_dim: int,
    default_freq_dim: int,
    adapter_config: dict[str, Any] | None = None,
) -> nn.Module:
    cfg = dict(adapter_config or {})
    adapter_type = str(cfg.get("type", "mlp")).strip().lower()
    freq_dim = int(cfg.get("freq_dim", default_freq_dim))

    if adapter_type in {"mlp", "width"}:
        if cfg.get("temporal") or cfg.get("temporal_spatial"):
            raise ValueError(
                f"Adapter type '{adapter_type}' does not accept `temporal` or "
                "`temporal_spatial` sub-configs."
            )
        return MLPResidualAdapter(
            action_dim=int(action_dim),
            freq_dim=freq_dim,
            hidden_dim=int(cfg.get("hidden_dim", 512)),
            depth=int(cfg.get("depth", 3)),
        )

    if adapter_type in {
        "temporal",
        "temporal_transformer",
        "temporal_spatial",
        "temporal_spatial_transformer",
    }:
        # Accept both legacy `temporal` and current `temporal_spatial` config names,
        # but route them through the same implemented adapter to avoid config landmines.
        temporal_cfg = dict(cfg.get("temporal_spatial", {}))
        legacy_temporal_cfg = dict(cfg.get("temporal", {}))
        if temporal_cfg and legacy_temporal_cfg:
            raise ValueError(
                "Use only one of `adapter.temporal_spatial` or `adapter.temporal`, not both."
            )
        if not temporal_cfg:
            temporal_cfg = legacy_temporal_cfg

        num_layers = int(temporal_cfg.get("num_layers", cfg.get("depth", 4)))
        if "depth" in cfg and "num_layers" in temporal_cfg:
            if int(cfg["depth"]) != num_layers:
                raise ValueError(
                    f"Conflicting adapter depth settings: depth={cfg['depth']} vs "
                    f"temporal.num_layers={temporal_cfg['num_layers']}"
                )

        hidden_dim = int(cfg.get("hidden_dim", temporal_cfg.get("hidden_dim", 1024)))
        if "hidden_dim" in temporal_cfg and int(temporal_cfg["hidden_dim"]) != hidden_dim:
            raise ValueError(
                f"Conflicting adapter hidden_dim settings: hidden_dim={cfg.get('hidden_dim')} vs "
                f"temporal.hidden_dim={temporal_cfg['hidden_dim']}"
            )

        num_heads = int(temporal_cfg.get("num_heads", 8))
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Temporal adapter requires hidden_dim % num_heads == 0, got "
                f"hidden_dim={hidden_dim}, num_heads={num_heads}"
            )

        return TemporalSpatialResidualAdapter(
            action_dim=int(action_dim),
            freq_dim=freq_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_mult=int(temporal_cfg.get("ffn_mult", 4)),
            dropout=float(temporal_cfg.get("dropout", 0.0)),
            max_horizon=int(temporal_cfg.get("max_horizon", 128)),
            local_kernel_size=int(temporal_cfg.get("local_kernel_size", 3)),
        )

    raise ValueError(f"Unsupported foresight adapter type: {adapter_type}")


def build_foresight_feature_adapter(
    *,
    hidden_dim: int,
    freq_dim: int,
    adapter_config: dict[str, Any] | None = None,
) -> nn.Module:
    cfg = dict(adapter_config or {})
    return FeatureResidualAdapter(
        hidden_dim=int(cfg.get("hidden_dim", hidden_dim)),
        freq_dim=int(cfg.get("freq_dim", freq_dim)),
        depth=int(cfg.get("depth", 2)),
        mlp_mult=int(cfg.get("mlp_mult", 2)),
    )
