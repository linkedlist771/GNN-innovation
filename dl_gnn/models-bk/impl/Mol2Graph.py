from typing import Final, Tuple
import torch.nn as nn
from torch import Tensor
import torch
from .ModUtils import CosineCutoff, Imod
from .Rbf import rbf_class_mapping

EPS = 1e-6


class EfDecay(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ea, ef):
        return ea, ea.unsqueeze(-1) * ef


class Mol2Graph(nn.Module):
    """
    Convert a molecule to a graph that GNNLF can process
    """

    def __init__(
        self,
        hid_dim: int,
        ef_dim: int,
        rbf: str,
        cutoff: float,
        max_z: int,
        ef_decay: bool = False,
        ln_emb: bool = False,
        **kwargs
    ):
        super().__init__()
        self.cutoff_fn = CosineCutoff(cutoff)
        self.rbf_fn = rbf_class_mapping[rbf](ef_dim, cutoff, **kwargs)
        self.atomic_number_embedding = nn.Embedding(max_z + 1, hid_dim, padding_idx=0)
        self.ef_decay = EfDecay() if ef_decay else Imod()
        self.atomic_number_layer_norm_embedding = (
            (
                nn.LayerNorm(
                    hid_dim,
                    # hid_dim 就是隐藏层的维度， 层归一化就是对这个隐藏层进行归一化， 输入和输出是一样的。
                    elementwise_affine=kwargs["ln_learnable"],
                )
            )
            if ln_emb
            else (nn.Identity())
        )

    def forward(self, atomic_numbers: Tensor, atomic_positions: Tensor):
        """
        Forward pass of the Mol2Graph module to process molecule data into graph representations.

        This function computes the embedding and spatial relationships of atoms in a molecule.
        It calculates atomic number embeddings, interatomic distances and their normalized vectors,
        edge features using radial basis function (RBF), and an atomic adjacency matrix with smooth cutoff.

        :param atomic_numbers: Tensor containing the atomic numbers of each atom in the molecule.
                               Shape: (batch_size, atomic_number)
        :param atomic_positions: Tensor containing the 3D coordinates of each atom in the molecule.
                                 Shape: (batch_size, atomic_number, 3)

        :return: A tuple of four elements:
                 - atomic_number_embedding: Embeddings of atomic numbers with layer normalization applied.
                                           Shape: (batch_size, atomic_number, hid_dim)
                 - atomic_adjacency_matrix: Smoothed adjacency matrix representing atomic connections.
                                           Shape: (batch_size, atomic_number, atomic_number)
                 - normalized_atom_position_distances: Normalized vectors representing interatomic distances.
                                                      Shape: (batch_size, atomic_number, atomic_number, 3)
                 - edge_features: Edge features computed using RBF, representing interatomic relationships.
                                 Shape: (batch_size, atomic_number, atomic_number, ef_dim)
        """

        epsilon = (
            1e-6  # 它通常在数学和计算领域中用作表示一个非常小的正数，接近于零但不为零
        )
        batch_size, atomic_number = (
            atomic_numbers.shape[0],
            atomic_numbers.shape[1],
        )  # batch size, number of atoms
        atomic_number_embedding = self.atomic_number_embedding(atomic_numbers)
        atomic_number_embedding = self.atomic_number_layer_norm_embedding(
            atomic_number_embedding
        )
        atom_position_differences = atomic_positions.unsqueeze(
            2
        ) - atomic_positions.unsqueeze(1)
        # batch_size x atomic_number x atomic_number x 3 意思是每个原子与其他原子的坐标差
        # 对于一个具体的元素 atom_position_differences[i, j, k, :]，
        # 这表示在batch第 i 个分子中，第 j 个原子相对于第 k 个原子的坐标差（一个三维向量）
        idx = torch.arange(atomic_number, device=atomic_number_embedding.device)
        atom_position_differences[:, idx, idx] = 1
        # 将原子中自己到自己的坐标差设置为1，这样在计算距离时，避免除以0
        atom_position_distances = torch.linalg.vector_norm(
            atom_position_differences, dim=-1
        )
        #  把距离的三维矩阵进行求模，得到距离， 所以其形状为 batch_size x atomic_number x atomic_number
        atom_position_distances = atom_position_distances.clone()
        normalized_atom_position_distances = atom_position_differences / (
            atom_position_distances.unsqueeze(-1) + epsilon
        )
        atom_position_distances[:, idx, idx] = 0  # remove self_loop
        normalized_atom_position_distances[:, idx, idx] = 0
        #  把三维矩阵的向量差进行归一化。 TODO: 其实距离不是一定要归一化的， 因为距离的大小也是有意义的，
        # TODO: 归一化后就丧失其大小的信息了。

        edge_features = self.rbf_fn(atom_position_distances.reshape(-1, 1)).reshape(
            batch_size, atomic_number, atomic_number, -1
        )
        #  通过RBF函数将距离转换成特征向量， 这里的特征向量的维度是ef_dim。
        #  换言之，这个就是领接矩阵边的特征向量。
        atomic_adjacency_matrix = self.cutoff_fn(atom_position_distances)
        #  通过领接矩阵来计算原子之间的邻接关系， 这里的领接矩阵不是0和1， 而是0到1的过渡， 通过余弦函数来光滑过渡。
        mask = atomic_numbers == 0
        mask = (mask).unsqueeze(2) + (mask).unsqueeze(1)
        atomic_adjacency_matrix[mask] = 0
        atomic_adjacency_matrix, edge_features = self.ef_decay(
            atomic_adjacency_matrix, edge_features
        )
        #  edge feature是否要通过decay来进行衰减， 这里的decay是通过领接矩阵来进行的。
        return (
            atomic_number_embedding,
            atomic_adjacency_matrix,
            normalized_atom_position_distances,
            edge_features,
        )
