import dgl
import numpy as np
import torch
import os
from typing import List, Tuple, Any, Optional, Union
from torch.utils.data import Dataset
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class BaseMD17Dataset(Dataset):

    def default_transform(self):
        pass

    def __init_from_data_path(self, data_path: str, **kwargs):
        if data_path.endswith("npz") or data_path.endswith("npy"):
            raw_data = np.load(data_path)
        else:
            raise NotImplementedError
        self.__init_from_raw_data(raw_data, **kwargs)

    def transform(self):
        if getattr(self, "_transform", None) is None:
            self.default_transform()
        else:
            _transform = getattr(self, "_transform")
            _transform()

    def __init_from_raw_data(self, raw_data: Any, **kwargs):
        energy, force, position, atomic_number = (
            raw_data["E"],
            raw_data["F"],
            raw_data["R"],
            raw_data["z"],
        )
        self.energy = energy
        self.force = force
        self.position = position
        self.atomic_number = atomic_number
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __init__(self, raw_data_or_data_path: Union[str, Any], **kwargs):
        if isinstance(raw_data_or_data_path, str):
            self.__init_from_data_path(raw_data_or_data_path, **kwargs)
        else:
            self.__init_from_raw_data(raw_data_or_data_path, **kwargs)
        self.transform()

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        # without transform first
        return self.energy[idx], self.force[idx], self.position[idx], self.atomic_number

    def __repr__(self):
        return (
            f"energy: {self.energy}\nforce: {self.force}\nposition: "
            f"{self.position}\natomic_number: {self.atomic_number}\n"
            f"energy shape: {self.energy.shape}\nforce shape: {self.force.shape}\n"
            f"position shape: {self.position.shape}\natomic_number shape: {self.atomic_number.shape}"
        )


@dataclass
class GraphSageMD17Dataset(BaseMD17Dataset):
    def __init__(self, **kwargs):
        super(GraphSageMD17Dataset, self).__init__(**kwargs)
        self.graphs = None
        self.number_of_atoms = self.position.shape[1]

    def build_directional_edge(self, idx: int = 0):
        """
        这里用于构建分子结构转换成有向图的有向边， 先采用最简单的方法来实现吧。
        Returns
        -------
        """
        # 像这样构建的话， 所有的图都是相同的， 不同的就是edge_feature.
        atomic_number_array = np.arange(self.number_of_atoms).astype(np.int32)
        u = []
        v = []
        for i in atomic_number_array:
            for j in atomic_number_array:
                if i != j:
                    u.append(i)
                    v.append(j)
        u = np.array(u)
        v = np.array(v)
        return u, v

    def build_node_features(self, idx: int):
        """
        这里的节点的特征可以选取为原子序数的embedding， 目前先省略
        Parameters
        ----------
        idx

        Returns
        -------

        """
        # 每个节点的embedding和其原子系数有关， 对于一个体系而言， 图中的每个节点的embedding都是相同的。
        # number_of_atoms \times embedding_length
        # 简单起见， 先用原子序数的embedding来表示节点的embedding
        _atomic_number = self.atomic_number
        _atomic_number = _atomic_number[:, np.newaxis].astype(np.float64)
        return _atomic_number

    def build_edge_features(self, idx: int):
        """
        这里选取两个原子序数之间的向量差作为特征
        Parameters
        ----------
        idx

        Returns
        -------

        """
        # edge weightes
        pos = self.position[idx]
        pos_diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        return pos_diff

    def to_dgl_homo_graph(self, idx: int = 0):
        u, v = self.build_directional_edge()
        graph = dgl.graph((u, v))
        edge_features = self.build_edge_features(idx)
        edge_features = torch.tensor(edge_features)
        node_features = self.build_node_features(idx)
        node_features = torch.tensor(node_features)
        # number_atoms x 3 => number_atoms x number_atoms x 3

        # 其实有 len(u) 这么多条边，所以, 由于每条边的特征是3维的位置向量差
        # 所以， edge feature 的形状为 len(u) \times 3, 其中
        # 每一个的 u[i] -> v[j] 的特征由 edge_features[i, j] 给出。
        edge_features = edge_features[u, v, :]  # 420 \times 3 shape, that is correct!
        # 如果 要把形状变成420 \times1 1 呢， 就是最后一个三维向量取一个模方就行了。
        edge_features = torch.norm(edge_features, dim=1, keepdim=True)

        graph.edata["feat"] = (
            edge_features  # that should be number_of_atoms \times embedding length.
        )
        graph.ndata["feat"] = (
            node_features  # that should be number_of_atoms \times embedding length.
        )
        # force = self.force[idx]
        # force = torch.tensor(force)
        # energy = self.energy[idx]
        # energy = torch.tensor(energy)
        return graph

    def build_graphs(self, processed_graph_path: Optional[str] = None):
        if processed_graph_path is not None and os.path.exists(processed_graph_path):
            raise NotImplementedError

        graphs = []
        for idx in tqdm(range(len(self)), desc="Building graphs"):
            graph = self.to_dgl_homo_graph(idx)
            graphs.append(graph)
        self.graphs = graphs

    def __getitem__(self, idx: int):
        if self.graphs is None:
            raise ValueError("you should `build graphs` first!")
        force = self.force[idx]
        force = torch.tensor(force)
        energy = self.energy[idx]
        energy = torch.tensor(energy)
        return self.graphs[idx], force, energy

    def default_transform(self):
        pass

    def __repr__(self):
        return (
            f"energy: {self.energy}\nforce: {self.force}\nposition: "
            f"{self.position}\natomic_number: {self.atomic_number}\n"
            f"energy shape: {self.energy.shape}\nforce shape: {self.force.shape}\n"
            f"position shape: {self.position.shape}\natomic_number shape: {self.atomic_number.shape}"
        )

    def save_graphs(self, save_path):
        raise NotImplementedError


def main():
    md17_aspirin_path = r"C:\Users\23174\Desktop\GitHub Project\GitHubProjectBigData\GNN-Molecular-Project\MD17\raw\md17_aspirin.npz"
    graph_sage_md17_dataset = GraphSageMD17Dataset(
        raw_data_or_data_path=md17_aspirin_path
    )
    g = graph_sage_md17_dataset.to_dgl_homo_graph()
    print(g)


if __name__ == "__main__":
    main()
