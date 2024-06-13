from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import SiLU
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from dl_gnn.models.visnet.models.utils import (
    CosineCutoff,
    Distance,
    EdgeEmbedding,
    NeighborEmbedding,
    Sphere,
    VecLayerNorm,
    act_class_mapping,
    rbf_class_mapping,
)


class GNNLFAttentionMessage(nn.Module):
    def __init__(
            self,
            num_heads,
            hidden_channels,
            # activation,
            # attn_activation,
            # cutoff,
            last_layer=False,
    ):
        super(GNNLFAttentionMessage, self).__init__()
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer
        self.layernorm = nn.LayerNorm(hidden_channels)
        # self.vec_layernorm = VecLayerNorm(
        #     hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type
        # )
        self.act = SiLU()
        self.attn_activation = SiLU()

        # self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        # self.vec_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)

        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

    def forward(self, s, ef_mask):
        """
        :param s: batch_size x atomic_number x hid_dim    => Node Features
        :param ef_mask: batch_size x atomic_number x atomic_number x hid_dim  => Edge Features
        :return:
        """
        batch_size, atomic_number, hid_dim = s.shape
        x = s

        vec = ef_mask
        x = self.layernorm(x)
        vec = self.layernorm(vec)  # 这里后面调整一下, 只是调整一下hidden_channels, 只是把最后一个维度的hid_dim 变成hidden_channels * num_heads
        q = self.q_proj(x).view(batch_size, atomic_number, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, atomic_number, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, atomic_number, self.num_heads, self.head_dim)
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        x, vec_out = self.propagate(
            q,
            k,
            v,
            vec,
            size=None,
        )

        # x shpae: torch.Size([6, 21, 256])

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=-1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        # if not self.last_layer:
        #     # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
        #     df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
        #     return dx, dvec, df_ij
        # else:
        return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j):
        attn = (q_i * k_j).sum(dim=-1)
        attn = self.attn_activation(attn)
        # 完全理解不了这里是怎么算的，他就是这样的喽， 不管了。

        v_j = (v_j * attn.unsqueeze(-1))  # batch_size x atomic_number x hid_dim
        # v_j shape torch.Size([6, 21, 8, 32])
        v_j = v_j.view(v_j.size(0), v_j.size(1), -1)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=-1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1)

        return v_j, vec_j

    # def edge_update(self, vec_i, vec_j, d_ij, f_ij):
    #     w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
    #     w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
    #     w_dot = (w1 * w2).sum(dim=1)
    #     df_ij = self.act(self.f_proj(f_ij)) * w_dot
    #     return df_ij

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
        agger_method = "mean"
        x, vec = features
        node_dim = 0
        x = scatter(x, index, dim=node_dim, dim_size=dim_size, reduce=agger_method)
        vec = scatter(vec, index, dim=node_dim, dim_size=dim_size, reduce=agger_method)  # TODO: 修改聚合里面的求和为mean防止数值梯度爆炸

        return x, vec

    def propagate(self, q, k, v, vec, size):
        v_j, vec_j = self.message(q, k, v, vec)
        # x, vec_out = self.aggregate((v_j, vec_j), size[1], size[0], size[1])
        return v_j, vec_j

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
        return inputs