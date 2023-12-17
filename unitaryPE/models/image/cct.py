import torch
from torch import Tensor
from torch.nn import Module, Conv2d, Linear, MaxPool2d, Sequential, ReLU

from ...neural.encoder import Encoder
from ...neural.position import UnitaryGrid, SinusoidalGrid
from ...neural.attention import multihead_atn_fn


class CCT(Module):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            num_heads: int,
            in_channels: int,
            kernel_size: tuple[int, int],
            num_classes: int,
            mlp_ratio: int):
        super(CCT, self).__init__()
        self.patch_embed = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=(1, 1),
                bias=False
            ),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size))
        self.encoder = Encoder(
            num_heads=num_heads,
            num_layers=num_layers,
            dim=dim,
            dropout_rate=0.1,
            weight_dropout=0.1,
            activation='GELU',
            drop_path=True,
            mlp_ratio=mlp_ratio)
        self.positional_encoder = UnitaryGrid(num_axes=2, num_heads=num_heads, dim=dim//(num_heads*2))
        self.pooler = Linear(dim, 1)
        self.fc = Linear(dim, num_classes)

    def forward(self, pixel_values: Tensor) -> Tensor:
        patch_values = self.patch_embed(pixel_values)
        h, w = patch_values.shape[-2], patch_values.shape[-2]
        patch_values = patch_values.permute(0, 2, 3, 1).flatten(1, 2)
        x_pos = torch.arange(0, h, device=patch_values.device).unsqueeze(-1).expand(h, w).flatten()[None]
        y_pos = torch.arange(0, w, device=patch_values.device).unsqueeze(0).expand(h, w).flatten()[None]
        self.positional_encoder.precompute(max(h, w))
        x_maps, y_maps = self.positional_encoder.forward(x_pos, y_pos)
        coords = torch.cat((x_pos, y_pos), dim=0)
        dists = ((coords[:, None] - coords[:, :, None]) ** 2).sum(0).sqrt().floor()[None, :, :, None, None]
        atn_fn = self.positional_encoder.adjust_attention((x_maps, y_maps), (x_maps, y_maps), (0.98 ** dists, True))
        patch_values = self.encoder.forward(
            encoder_input=patch_values,
            encoder_mask=None,  # type: ignore
            atn_fn=atn_fn)
        gate = self.pooler(patch_values).softmax(dim=1)
        aggr = (gate * patch_values).sum(dim=1)
        return self.fc(aggr)


class SinusoidalCCT(Module):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            num_heads: int,
            in_channels: int,
            kernel_size: tuple[int, int],
            num_classes: int,
            mlp_ratio: int):
        super(SinusoidalCCT, self).__init__()
        self.patch_embed = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=(1, 1),
                bias=False
            ),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size))
        self.encoder = Encoder(
            num_heads=num_heads,
            num_layers=num_layers,
            dim=dim,
            dropout_rate=0.1,
            weight_dropout=0.1,
            activation='GELU',
            drop_path=True,
            mlp_ratio=mlp_ratio)
        self.positional_encoder = SinusoidalGrid(dim=dim)
        self.pooler = Linear(dim, 1)
        self.fc = Linear(dim, num_classes)

    def forward(self, pixel_values: Tensor) -> Tensor:
        patch_values = self.patch_embed(pixel_values)
        patch_values = patch_values.permute(0, 2, 3, 1).flatten(1, 2)
        positions = torch.arange(0, patch_values.shape[1], device=patch_values.device)[None]
        pos_emb = self.positional_encoder.forward(positions)
        patch_values = patch_values + pos_emb
        patch_values = self.encoder.forward(
            encoder_input=patch_values,
            encoder_mask=None,  # type: ignore
            atn_fn=multihead_atn_fn)
        gate = self.pooler(patch_values).softmax(dim=1)
        aggr = (gate * patch_values).sum(dim=1)
        return self.fc(aggr)
