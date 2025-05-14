import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GPSConv, global_add_pool

class GraphGPS(nn.Module):
    """Full GraphGPS block:
       * optional LapPE + RWPE (dimension = 8 + 16)
       * local GINConv + global sparse attention (GPSConv)
    """
    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 heads: int = 4,
                 pe_dim: int = 24):                    # 8 + 16
        super().__init__()

        # project raw features (+ optional PE) → hidden dim
        self.input_proj = nn.Linear(in_channels + pe_dim, hidden_channels)

        def local_gin(dim):
            return GINConv(
                nn.Sequential(nn.Linear(dim, dim),
                              nn.ReLU(),
                              nn.Linear(dim, dim)))

        self.convs = nn.ModuleList([
            GPSConv(channels=hidden_channels,
                    conv=local_gin(hidden_channels),
                    heads=heads)
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU())
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.pe_dim = pe_dim                                   # store for pad

    # ------------------------------------------------------------------
    # forward: works w/ or w/o positional encodings --------------------
    # ------------------------------------------------------------------
    def forward(self, x, edge_index, batch,
                lap_pe: torch.Tensor = None,
                rw_pe:  torch.Tensor = None):
        # cat positional encodings if provided; else pad zeros
        if lap_pe is None or rw_pe is None:
            pe = x.new_zeros((x.size(0), self.pe_dim))
        else:
            pe = torch.cat([lap_pe, rw_pe], dim=-1)
        x = torch.cat([x, pe], dim=-1)
        x = self.input_proj(x)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))          # GPS block
        x = global_add_pool(x, batch)                # graph-level pooling
        return self.classifier(self.readout(x))