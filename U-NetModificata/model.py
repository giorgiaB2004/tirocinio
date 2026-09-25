from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # GroupNorm remains valid for batch size 1 and a 1x1 feature map.
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        x = self.activation(self.norm(self.conv(self.pool(x))))
        return F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class ASPPConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        *,
        separable: bool,
    ) -> None:
        convolution: nn.Module
        if separable:
            convolution = SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        else:
            convolution = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        super().__init__(
            convolution,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        out_channels: int,
        *,
        atrous_rates: Sequence[int] = (2, 4, 6),
        separable: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        rates = tuple(int(rate) for rate in atrous_rates)
        if len(rates) != 3 or any(rate <= 0 for rate in rates):
            raise ValueError("atrous_rates must contain exactly three positive values")

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                ),
                ASPPConv(
                    in_channels,
                    branch_channels,
                    rates[0],
                    separable=separable,
                ),
                ASPPConv(
                    in_channels,
                    branch_channels,
                    rates[1],
                    separable=separable,
                ),
                ASPPConv(
                    in_channels,
                    branch_channels,
                    rates[2],
                    separable=separable,
                ),
                ASPPPooling(in_channels, branch_channels),
            ]
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                5 * branch_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.convs], dim=1))


class ClassAwareAttentionGate(nn.Module):
    def __init__(
        self,
        F_g: int,
        F_l: int,
        F_int: int,
        *,
        num_classes: int = 3,
        class_weights: Sequence[float] | None = None,
        min_attention: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 <= min_attention <= 1.0:
            raise ValueError("min_attention must be in [0, 1]")

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # These logits receive an auxiliary CE loss in train.py.
        self.psi = nn.Conv2d(F_int, num_classes, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.min_attention = float(min_attention)

        weights = torch.ones(num_classes, dtype=torch.float32)
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.numel() != num_classes:
            raise ValueError(
                f"Expected {num_classes} class weights, received {weights.numel()}"
            )
        if torch.any(weights < 0) or torch.all(weights == 0):
            raise ValueError("Class weights must be non-negative and not all zero")

        # Equal weights become all ones, making the default gate an identity gate.
        weights = weights / weights.max().clamp_min(1e-8)
        self.register_buffer("class_weights", weights.view(1, num_classes, 1, 1))

    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gating = self.W_g(g)
        if gating.shape[-2:] != x.shape[-2:]:
            gating = F.interpolate(
                gating,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        skip_features = self.W_x(x)
        class_logits = self.psi(self.relu(gating + skip_features))
        class_probabilities = torch.softmax(class_logits, dim=1)

        attention = (class_probabilities * self.class_weights).sum(
            dim=1,
            keepdim=True,
        )
        attention = self.min_attention + (1.0 - self.min_attention) * attention
        return x * attention, class_logits


class DualAttentionGate(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive")
        hidden_channels = max(1, in_channels // reduction_ratio)

        self.avg_pool_channel = nn.AdaptiveAvgPool2d(1)
        self.max_pool_channel = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attention = torch.sigmoid(
            self.mlp(self.avg_pool_channel(x))
            + self.mlp(self.max_pool_channel(x))
        )
        x = x * channel_attention

        spatial_input = torch.cat(
            [
                torch.mean(x, dim=1, keepdim=True),
                torch.amax(x, dim=1, keepdim=True),
            ],
            dim=1,
        )
        spatial_attention = torch.sigmoid(self.conv_spatial(spatial_input))
        return x * spatial_attention


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.conv = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(inputs)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        attention_type: str = "none",
        num_classes: int = 3,
        class_weights: Sequence[float] | None = None,
        dropout_rate: float = 0.0,
        min_attention: float = 0.25,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.attention_type = attention_type

        self.class_aware_att: ClassAwareAttentionGate | None = None
        self.dual_att: DualAttentionGate | None = None

        if attention_type in {"class_aware", "both"}:
            self.class_aware_att = ClassAwareAttentionGate(
                F_g=out_channels,
                F_l=out_channels,
                F_int=max(1, out_channels // 2),
                num_classes=num_classes,
                class_weights=class_weights,
                min_attention=min_attention,
            )
        if attention_type in {"dual", "both"}:
            self.dual_att = DualAttentionGate(out_channels)

        self.conv = ConvBlock(2 * out_channels, out_channels, dropout_rate)

    def forward(
        self,
        inputs: torch.Tensor,
        skip: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.up(inputs)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        skip_processed = skip
        class_logits: torch.Tensor | None = None

        if self.dual_att is not None:
            skip_processed = self.dual_att(skip_processed)
        if self.class_aware_att is not None:
            skip_processed, class_logits = self.class_aware_att(
                g=x,
                x=skip_processed,
            )

        x = self.conv(torch.cat([x, skip_processed], dim=1))
        return x, class_logits


class UNet(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 3,
        in_channels: int = 4,
        attention_type: str | None = "none",
        class_weights: Sequence[float] | None = None,
        dropout_rate: float = 0.3,
        use_aspp: bool = False,
        base_channels: int = 64,
        aspp_branch_channels: int = 256,
        aspp_rates: Sequence[int] = (2, 4, 6),
        aspp_separable: bool = True,
        min_attention: float = 0.25,
    ) -> None:
        super().__init__()
        attention_type = "none" if attention_type is None else attention_type
        valid_attention = {"none", "class_aware", "dual", "both"}
        if attention_type not in valid_attention:
            raise ValueError(
                f"attention_type must be one of {sorted(valid_attention)}, "
                f"received {attention_type!r}"
            )
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if in_channels <= 0 or base_channels <= 0:
            raise ValueError("in_channels and base_channels must be positive")

        channels = [
            base_channels,
            2 * base_channels,
            4 * base_channels,
            8 * base_channels,
            16 * base_channels,
        ]

        self.e1 = EncoderBlock(in_channels, channels[0])
        self.e2 = EncoderBlock(channels[0], channels[1])
        self.e3 = EncoderBlock(channels[1], channels[2])
        self.e4 = EncoderBlock(channels[2], channels[3])

        self.b = ConvBlock(channels[3], channels[4], dropout_rate)
        self.use_aspp = bool(use_aspp)
        self.aspp: ASPP | None = None
        if self.use_aspp:
            self.aspp = ASPP(
                in_channels=channels[4],
                branch_channels=aspp_branch_channels,
                out_channels=channels[4],
                atrous_rates=aspp_rates,
                separable=aspp_separable,
                dropout=dropout_rate,
            )

        decoder_common = {
            "attention_type": attention_type,
            "num_classes": num_classes,
            "class_weights": class_weights,
            "min_attention": min_attention,
        }
        self.d1 = DecoderBlock(
            channels[4],
            channels[3],
            dropout_rate=dropout_rate,
            **decoder_common,
        )
        self.d2 = DecoderBlock(
            channels[3],
            channels[2],
            dropout_rate=dropout_rate,
            **decoder_common,
        )
        self.d3 = DecoderBlock(
            channels[2],
            channels[1],
            dropout_rate=0.0,
            **decoder_common,
        )
        self.d4 = DecoderBlock(
            channels[1],
            channels[0],
            dropout_rate=0.0,
            **decoder_common,
        )
        self.outputs = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        self.config = {
            "num_classes": num_classes,
            "in_channels": in_channels,
            "attention_type": attention_type,
            "class_weights": (
                list(class_weights) if class_weights is not None else None
            ),
            "dropout_rate": float(dropout_rate),
            "use_aspp": bool(use_aspp),
            "base_channels": int(base_channels),
            "aspp_branch_channels": int(aspp_branch_channels),
            "aspp_rates": list(aspp_rates),
            "aspp_separable": bool(aspp_separable),
            "min_attention": float(min_attention),
        }

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        bottleneck = self.b(p4)
        if self.aspp is not None:
            bottleneck = self.aspp(bottleneck)

        attention_logits: list[torch.Tensor] = []
        d1, a1 = self.d1(bottleneck, s4)
        d2, a2 = self.d2(d1, s3)
        d3, a3 = self.d3(d2, s2)
        d4, a4 = self.d4(d3, s1)
        for logits in (a1, a2, a3, a4):
            if logits is not None:
                attention_logits.append(logits)

        outputs = self.outputs(d4)
        return outputs, bottleneck, tuple(attention_logits)


def build_unet(
    num_classes: int = 3,
    attention_type: str | None = "none",
    class_weights: Sequence[float] | None = None,
    dropout_rate: float = 0.3,
    use_aspp: bool = False,
    *,
    in_channels: int = 4,
    base_channels: int = 64,
    aspp_branch_channels: int = 256,
    aspp_rates: Sequence[int] = (2, 4, 6),
    aspp_separable: bool = True,
    min_attention: float = 0.25,
) -> UNet:
    return UNet(
        num_classes=num_classes,
        in_channels=in_channels,
        attention_type=attention_type,
        class_weights=class_weights,
        dropout_rate=dropout_rate,
        use_aspp=use_aspp,
        base_channels=base_channels,
        aspp_branch_channels=aspp_branch_channels,
        aspp_rates=aspp_rates,
        aspp_separable=aspp_separable,
        min_attention=min_attention,
    )


# Backward-compatible aliases for external imports.
conv_block = ConvBlock
encoder_block = EncoderBlock
decoder_block = DecoderBlock


if __name__ == "__main__":
    model = build_unet(
        num_classes=3,
        attention_type="dual",
        class_weights=[1.0, 2.21, 5.42],
    )
    sample = torch.randn(1, 4, 256, 256)
    logits, bottleneck, attention_logits = model(sample)
    print("Logits:", tuple(logits.shape))
    print("Bottleneck:", tuple(bottleneck.shape))
    print("Attention maps:", [tuple(item.shape) for item in attention_logits])
    print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))

    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("ptflops is not installed; FLOPs were not computed.")
    else:
        original_forward = model.forward
        model.forward = lambda x: original_forward(x)[0]  # type: ignore[method-assign]
        flops, params = get_model_complexity_info(
            model,
            input_res=(4, 256, 256),
            as_strings=True,
            print_per_layer_stat=False,
        )
        print("FLOPs:", flops)
        print("Parameters (ptflops):", params)
