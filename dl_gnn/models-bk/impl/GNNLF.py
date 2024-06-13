# GNN-LF model for MD17 dataset.
from typing import Final
import torch.nn as nn
import torch
from torch.nn import ModuleList
from .Mol2Graph import Mol2Graph
from .Utils import innerprod
from torch import Tensor


class GNNLF(torch.nn.Module):
    ev_decay: Final[bool]
    add_ef2dir: Final[bool]
    use_dir1: Final[bool]  # whether to use another coordinate projection
    use_dir2: Final[bool]  # whether to use coordinate projection
    use_dir3: Final[bool]  # whether to use frame-frame projection
    global_frame: Final[bool]  # for ablation study. whether to use global frame
    no_filter_decomp: Final[
        bool
    ]  # for ablation study. whether to use filter decomposition trick
    no_share_filter: Final[bool]  # for ablation study. whether to share filter

    def __init__(
        self,
        hid_dim: int,
        num_mplayer: int,
        ef_dim: int,
        y_mean: float,
        y_std: float,
        global_y_mean: float,
        ev_decay: bool,
        add_ef2dir: bool,
        global_frame: bool = False,
        use_dir1: bool = False,
        use_dir2: bool = True,
        use_dir3: bool = True,
        no_filter_decomp: bool = False,
        no_share_filter: bool = False,
        **kwargs
    ):
        super().__init__()
        self.no_share_filter = no_share_filter
        self.global_frame = global_frame
        self.no_filter_decomp = no_filter_decomp
        self.add_ef2dir = add_ef2dir
        kwargs["ln_learnable"] = False
        kwargs["act"] = nn.SiLU(inplace=True)
        self.mol2graph = Mol2Graph(hid_dim, ef_dim, **kwargs)
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float))
        self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float))
        self.register_buffer(
            "global_y_mean", torch.tensor(global_y_mean, dtype=torch.float64)
        )

        self.neighbor_emb = NeighborEmb(hid_dim, **kwargs)
        self.s2v = CFConvS2V(hid_dim, **kwargs)
        self.q_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.k_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.interactions = ModuleList(
            [DirCFConv(hid_dim, **kwargs) for _ in range(num_mplayer)]
        )
        self.output_module = nn.Linear(hid_dim, 1)
        self.ef_proj = nn.Sequential(
            nn.Linear(ef_dim, hid_dim),
            (
                nn.SiLU(inplace=True)
                if kwargs[
                    "ef2mask_tailact"
                ]  # inplace=True 参数表示在进行前向计算时，直接在输入数据上进行原地操作以节省内存
                else nn.Identity()
            ),
        )
        dir_dim = (
            hid_dim * (use_dir1 + use_dir2 + use_dir3)
            + (self.add_ef2dir or self.no_filter_decomp) * ef_dim
        )
        self.dir_proj = nn.Sequential(
            nn.Linear(dir_dim, dir_dim),
            nn.SiLU(inplace=True),
            nn.Linear(dir_dim, hid_dim),
            nn.SiLU(inplace=True) if kwargs["dir2mask_tailact"] else nn.Identity(),
        )
        self.ev_decay = ev_decay
        self.use_dir1 = use_dir1
        self.use_dir2 = use_dir2
        self.use_dir3 = use_dir3
        if self.no_share_filter:
            self.ef_projections = ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(ef_dim, hid_dim),
                        (
                            nn.SiLU(inplace=True)
                            if kwargs["ef2mask_tailact"]
                            else nn.Identity()
                        ),
                    )
                    for i in range(num_mplayer)
                ]
            )
            self.dir_projections = ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dir_dim, dir_dim),
                        nn.SiLU(inplace=True),
                        nn.Linear(dir_dim, hid_dim),
                        (
                            nn.SiLU(inplace=True)
                            if kwargs["dir2mask_tailact"]
                            else nn.Identity()
                        ),
                    )
                    for i in range(num_mplayer)
                ]
            )

    def forward(self, atomic_numbers: Tensor, atomic_positions: Tensor):
        """
        神经网络前向传播
        :param atomic_numbers: the atomic number of each atom in the molecule, shape: batch_size x atomic_number
        :param atomic_positions:  the coordinate of each atom in the molecule, shape: batch_size x atomic_number x 3
        :return:
         - atomic_number_embedding: Embeddings of atomic numbers with layer normalization applied.
                                           Shape: (batch_size, atomic_number, hid_dim)
         - atomic_adjacency_matrix: Smoothed adjacency matrix representing atomic connections.
                                           Shape: (batch_size, atomic_number, atomic_number)
         - normalized_atom_position_distances: Normalized vectors representing interatomic distances.
                                                      Shape: (batch_size, atomic_number, atomic_number, 3)
         - edge_features: Edge features computed using RBF, representing interatomic relationships.
                                 Shape: (batch_size, atomic_number, atomic_number, ef_dim)
        """
        (
            atomic_number_embedding,
            atomic_adjacency_matrix,
            normalized_atom_position_distances,
            edge_features,
        ) = self.mol2graph(atomic_numbers, atomic_positions)
        mask = self.ef_proj(edge_features) * atomic_adjacency_matrix.unsqueeze(-1)
        #  unsqueeze(-1) 表示在最后一个维度上增加一个维度， 这里是增加一个维度， 使得其与mask的维度一致。
        s = self.neighbor_emb(atomic_numbers, atomic_number_embedding, mask)
        v = self.s2v(s, normalized_atom_position_distances, mask)
        if self.global_frame:
            v = torch.sum(v, dim=1, keepdim=True).expand(-1, s.shape[1], -1, -1)
        dirs = []
        if self.use_dir1:
            dir1 = innerprod(
                v.unsqueeze(1), normalized_atom_position_distances.unsqueeze(-1)
            )
            dirs.append(dir1)
        if self.use_dir2:
            dir2 = innerprod(
                v.unsqueeze(2), normalized_atom_position_distances.unsqueeze(-1)
            )
            dirs.append(dir2)
        if self.use_dir3:
            dir3 = innerprod(self.q_proj(v).unsqueeze(1), self.k_proj(v).unsqueeze(2))
            dirs.append(dir3)
        dir = torch.cat(dirs, dim=-1)
        if self.ev_decay:
            dir = dir * atomic_adjacency_matrix.unsqueeze(-1)
        if self.add_ef2dir or self.no_filter_decomp:
            dir = torch.cat((dir, edge_features), dim=-1)
        if self.no_filter_decomp:
            mask = self.dir_proj(dir)
        else:
            mask = mask * self.dir_proj(dir)
        for layer, interaction in enumerate(self.interactions):
            if self.no_share_filter:
                mask = self.ef_projections[layer](edge_features) * self.dir_projections[
                    layer
                ](dir)
            s = interaction(s, mask) + s
        s[atomic_numbers == 0] = 0
        s = torch.sum(s, dim=1)
        s = self.output_module(s)
        s = s * self.y_std + self.y_mean
        # print(f"y_std: {self.y_std}, y_mean: {self.y_mean}")
        #  without this: Training (Epoch 42/6000): 100%|██████████| 8/8 [00:01<00:00,  5.91it/s, energy_loss=733, force_loss=25.8]
        #  whith this: Training (Epoch 42/6000): 100%|██████████| 8/8 [00:01<00:00,  6.00it/s, energy_loss=280, force_loss=26.8]
        #  seems that work for force loss, 但是这里面的y_std 和 y_mean 并没有改变。 其实也没必要改变， 因为这里的y_std 和 y_mean。

        return s


class CFConv(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, s, mask):
        """
        s (B, N, hid_dim)
        v (B, N, 3, hid_dim)
        ea (B, N, N)
        ef (B, N, N, ef_dim)
        ev (B, N, N, 3)
        """
        s = mask * s.unsqueeze(1)
        s = torch.sum(s, dim=2)
        return s


class DirCFConv(nn.Module):

    def __init__(
        self,
        hid_dim: int,
        ln_lin1: bool,
        lin1_tailact: bool,
        nolin1: bool = False,
        **kwargs
    ):
        super().__init__()
        self.lin1 = (
            nn.Identity()
            if nolin1
            else nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                (
                    nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"])
                    if ln_lin1
                    else nn.Identity()
                ),
                kwargs["act"] if lin1_tailact else nn.Identity(),
            )
        )

    def forward(self, s, ef_mask):
        """
        s (B, N, hid_dim)
        v (B, N, 3, hid_dim)
        ea (B, N, N)
        ef (B, N, N, ef_dim)
        ev (B, N, N, 3)
        """
        s = torch.sum(ef_mask * self.lin1(s).unsqueeze(1), dim=2)
        return s


class NeighborEmb(nn.Module):

    def __init__(self, hid_dim: int, max_z: int, ln_emb: bool, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(max_z, hid_dim, padding_idx=0)
        self.conv = CFConv()
        self.ln_emb = (
            nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"])
            if ln_emb
            else nn.Identity()
        )

    def forward(self, z, s, mask):
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.conv(s_neighbors, mask)
        s = s + s_neighbors
        return s


class CFConvS2V(nn.Module):

    def __init__(
        self,
        hid_dim: int,
        ln_s2v: bool,
        lin1_tailact: bool,
        nolin1: bool = False,
        **kwargs
    ):
        super().__init__()
        self.lin1 = (
            nn.Identity()
            if nolin1
            else nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                (
                    nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"])
                    if ln_s2v
                    else nn.Identity()
                ),
                kwargs["act"] if lin1_tailact else nn.Identity(),
            )
        )

    def forward(self, s, ev, mask):
        """
        s (B, N, hid_dim)
        v (B, N, 3, hid_dim)
        ea (B, N, N)
        ef (B, N, N, ef_dim)
        ev (B, N, N, 3)
        """
        s = self.lin1(s)
        s = s.unsqueeze(1) * mask
        v = s.unsqueeze(3) * ev.unsqueeze(-1)
        v = torch.sum(v, dim=2)
        return v
