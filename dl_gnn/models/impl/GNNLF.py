# GNN-LF model for MD17 dataset.
from typing import Final
import torch.nn as nn
import torch
from torch.nn import ModuleList
from dl_gnn.models.impl.Mol2Graph import Mol2Graph
from dl_gnn.models.impl.Utils import innerprod
from torch import Tensor

from dl_gnn.models.impl.message_passing import GNNLFAttentionMessage
from dl_gnn.models.kan_models.model import KanOutPutModule
from dl_gnn.models.visnet.models import output_modules
from dl_gnn.models.visnet.models.utils import Sphere, VecLayerNorm
from dl_gnn.models.visnet.models.visnet_block import ViS_MP, ViS_MP_Vertex_Edge


class CFConv(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, s, mask):
        """
        :param s: batch_size x atomic_number x hid_dim
        :param mask: batch_size x atomic_number x atomic_number x hid_dim
        :return:
        """
        unsqueezed_s = s.unsqueeze(1)
        # unsqueezed_s: batch_size x 1 x atomic_number x hid_dim
        s = mask * unsqueezed_s
        # s: batch_size x atomic_number x atomic_number x hid_dim
        s = torch.sum(s, dim=2)
        # s: batch_size x atomic_number x hid_dim
        return s


class DirCFConv(nn.Module):

    def __init__(
        self,
        hid_dim: int,
        ln_lin1: bool,
        add_activation_to_lin1: bool,
        no_activation_to_lin1: bool = False,
        **kwargs
    ):
        super().__init__()
        self.lin1 = (
            nn.Identity()
            if no_activation_to_lin1
            else nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                (
                    nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"])
                    if ln_lin1
                    else nn.Identity()
                ),
                kwargs["act"] if add_activation_to_lin1 else nn.Identity(),
            )
        )

    def forward(self, s, ef_mask):
        """
        :param s: batch_size x atomic_number x hid_dim
        :param ef_mask: batch_size x atomic_number x atomic_number x hid_dim
        :return:
        """
        s = ef_mask * self.lin1(s).unsqueeze(1)
        # s: batch_size x atomic_number x atomic_number x hid_dim
        s = torch.sum(s, dim=2)
        #  s: batch_size x atomic_number x hid_dim
        return s




class CFConvS2V(nn.Module):

    def __init__(
        self,
        hid_dim: int,
        ln_s2v: bool,
        add_activation_to_lin1: bool,
        no_activation_to_lin1: bool = False,
        **kwargs
    ):
        super().__init__()
        self.lin1 = (
            nn.Identity()
            if no_activation_to_lin1
            else nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                (
                    nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"])
                    if ln_s2v
                    else nn.Identity()
                ),
                kwargs["act"] if add_activation_to_lin1 else nn.Identity(),
            )
        )

    def forward(self, s, ev, mask):
        """

        :param s: batch_size x atomic_number x hid_dim
        :param ev: batch_size x atomic_number x atomic_number x 3
        :param mask: batch_size x atomic_number x atomic_number x hid_dim
        :return:
        """
        s = self.lin1(s)  #
        # s: batch_size x atomic_number x hid_dim
        s = s.unsqueeze(1) * mask
        # s: batch_size x atomic_number x atomic_number x hid_dim
        v = s.unsqueeze(3) * ev.unsqueeze(-1)
        # v: batch_size x atomic_number x 3 x hid_dim
        v = torch.sum(v, dim=2)
        return v


class NeighborEmb(nn.Module):

    def __init__(self, hid_dim: int, max_z: int, ln_emb: bool, **kwargs):
        """
        # nn.Embedding是一种特殊的神经网络层，用于将离散的类别型输入（在这里是原子编号）转换为连续的向量表示。
        # 这个向量表示可以捕捉到输入间的复杂关系，这些关系不能通过输入的数值本身直接表示。
        # 在这个例子中，我们创建了一个nn.Embedding(max_z, hid_dim)层，其中max_z是原子编号的最大值（在这里设为20），
        # hid_dim是嵌入向量的维度。这将创建一个max_z x hid_dim的嵌入矩阵，每一行都是一个hid_dim维度的向量，
        # 对应一个原子编号的嵌入表示。这个嵌入矩阵在模型训练过程中会被学习和更新，以更好地捕捉原子间的关系。
        # 例如，如果我们有一个原子编号为5的输入，我们可以通过查找嵌入矩阵的第5行来获取这个原子的向量表示。
        # 注意，如果输入的原子编号大于max_z，那么nn.Embedding会抛出一个错误，因为它没有为这些原子编号预定义嵌入向量。
        # 因此，你需要根据你的数据设定一个合适的max_z值。
        # 也就是说这里的max_z只需要是原子序数中最大值即可。

        :param hid_dim:
        :param max_z:
        :param ln_emb:
        :param kwargs:
        """
        super().__init__()
        self.embedding = nn.Embedding(max_z, hid_dim, padding_idx=0)
        self.conv = CFConv()
        self.ln_emb = (
            nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"])
            if ln_emb
            else nn.Identity()
        )

    def forward(self, z, s, mask):
        # s: batch_size x atomic_number x hid_dim  => atomic embedding
        # mask: batch_size x atomic_number x atomic_number x hid_dim
        s_neighbors = self.ln_emb(self.embedding(z))
        # s_neighbors: batch_size x atomic_number x hid_dim
        s_neighbors = self.conv(s_neighbors, mask)
        # s_neighbors: batch_size x atomic_number x hid_dim
        # 其实就是把当前原子的邻居通过embedding转换成向量， 然后通过conv进行聚合。
        s = s + s_neighbors
        # 然后把聚合的结果和当前原子的向量进行相加。
        return s


class GNNLF(torch.nn.Module):
    ev_decay: Final[bool]
    add_ef2dir: Final[bool]
    use_dir1: Final[bool]  # whether to use another coordinate projection
    use_dir2: Final[bool]  # whether to use coordinate projection
    use_dir3: Final[bool]  # whether to use frame-frame projection
    global_frame: Final[bool]  # for ablation study. whether to use global frame
    no_filter_decomposition: Final[
        bool
    ]  # for ablation study. whether to use filter decomposition trick
    no_share_filter: Final[bool]  # for ablation study. whether to share filter

    def __init__(
        self,
        hid_dim: int,
        num_mplayer: int,
        ef_dim: int,
        ev_decay: bool,
        add_ef2dir: bool,
        global_frame: bool = False,
        use_dir1: bool = False,
        use_dir2: bool = True,
        use_dir3: bool = True,
        no_filter_decomposition: bool = False,
        no_share_filter: bool = False,
        colfnet_features: bool = False,
        use_drop_out: bool = False,
        use_visnet_output_modules: bool = False,
        use_kan_output_modules: bool = False,
        use_visnet_message_passing: bool = False,
        **kwargs
    ):
        super().__init__()
        atomic_number = kwargs.get("atomic_number", None)
        assert atomic_number is not None, "kwargs['atomic_number'] is required"
        self.no_share_filter = no_share_filter
        self.global_frame = global_frame
        self.no_filter_decomposition = no_filter_decomposition
        self.add_ef2dir = add_ef2dir
        kwargs["ln_learnable"] = False
        kwargs["act"] = nn.SiLU(inplace=True)
        self.mol2graph = Mol2Graph(hid_dim, ef_dim, **kwargs)
        self.neighbor_emb = NeighborEmb(hid_dim, **kwargs)
        self.s2v = CFConvS2V(hid_dim, **kwargs)
        self.hid_dim = hid_dim
        self.q_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.k_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.use_visnet_message_passing = use_visnet_message_passing
        self.out_norm = nn.LayerNorm(hid_dim)

        if self.use_visnet_message_passing:
            # 是否使用visnet的message passing
            # self.vis_mp_layer = ViS_MP(
            #     num_heads=8,
            #     hidden_channels=hid_dim,
            #     activation="silu",
            #     attn_activation="silu",
            #     cutoff=5.0,
            #     vecnorm_type="max_min",
            #     trainable_vecnorm=False,
            #     last_layer=False
            # )
            self.out_norm = nn.LayerNorm(hid_dim)
            self.vec_out_norm = VecLayerNorm(
                hid_dim, trainable=False, norm_type="max_min"
            )

            self.vis_mp_layers = nn.ModuleList()
            vis_mp_kwargs = dict(
                num_heads=8,
                hidden_channels=hid_dim,
                activation="silu",
                attn_activation="silu",
                cutoff=5.0,
                vecnorm_type="max_min",
                trainable_vecnorm=False,
            )
            for _ in range(num_mplayer - 1):
                layer = ViS_MP_Vertex_Edge(last_layer=False, **vis_mp_kwargs)#  .jittable()  # original VIS_MP
                self.vis_mp_layers.append(layer)
            self.vis_mp_layers.append(
                ViS_MP_Vertex_Edge(last_layer=True, **vis_mp_kwargs)# .jittable()
            )


        else:

            self.interactions = ModuleList(
                [DirCFConv(hid_dim, **kwargs) for _ in range(num_mplayer)]
            )
            # hid_dim = 256
            # num_heads = 8
            # message_net = GNNLFAttentionMessage(num_heads=num_heads, hidden_channels=hid_dim)
            # self.interactions = ModuleList(
            #     [GNNLFAttentionMessage(num_heads=8, hidden_channels=hid_dim) for _ in range(num_mplayer)]
            #     )

        # self.output_module = nn.Linear(hid_dim, 1)

        self.colfnet_features = colfnet_features
        self.colfnet_features_projection = nn.Sequential(
            nn.Linear(8, hid_dim), nn.SiLU(inplace=True)
        )
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
            + (self.add_ef2dir or self.no_filter_decomposition) * ef_dim
            + self.colfnet_features * hid_dim  # 8
        )
        # TODO: 在这里添加drop out层
        self.dir_proj = nn.Sequential(
            nn.Linear(dir_dim, dir_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5) if use_drop_out else nn.Identity(),
            nn.Linear(dir_dim, hid_dim),
            nn.SiLU(inplace=True) if kwargs["dir2mask_tailact"] else nn.Identity(),
            nn.Dropout(0.5) if use_drop_out else nn.Identity(),
            # 如果dir2mask_tailact为True，则在最后一个SiLU后也添加dropout层
        )
        # TODO: 先把里面的output_network加进去。
        # 先dropout = 0， 训练后得到模型的一些指标（比如: F1, Accuracy, AP）。比较train数据集和test数据集的指标。
        #
        # 过拟合：尝试下面的步骤。
        # 欠拟合：尝试调整模型的结构，暂时忽略下面步骤。
        # dropout设置成0
        # .4 - 0.6
        # 之间， 再次训练得到模型的一些指标。
        #
        # 如果过拟合明显好转，但指标也下降明显，可以尝试减少dropout（0.2）
        # 如果过拟合还是严重，增加dropout（0.2）
        # 重复上面的步骤多次，就可以找到理想的dropout值了。
        #
        # https: // zhuanlan.zhihu.com / p / 77609689
        # 注：dropout过大是容易欠拟合。

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
                    for _ in range(num_mplayer)
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
                    for _ in range(num_mplayer)
                ]
            )

        # 添加output modules
        self.use_visnet_output_modules = use_visnet_output_modules
        self.use_kan_output_modules = use_kan_output_modules
        if use_visnet_output_modules:
            self.equivalent_output_module = output_modules.EquivariantScalar(
                hidden_channels=hid_dim
            )
            self.equivalent_output_module.reset_parameters()
        elif self.use_kan_output_modules:

            self.kan_output_modules = KanOutPutModule(
                hid_dim, atomic_number, device=kwargs["device"]
            )
        else:
            self.output_module_1 = nn.Linear(hid_dim, 1)
            self.output_module_activation = nn.SiLU(inplace=True)
            self.output_module_2 = nn.Linear(atomic_number, 1)

        self.sphere = Sphere(l=2)
    def coordinates2localframe(
        self, position_batch_center, atomic_adjacency_matrix, norm_diff=True
    ):
        batch_size, num_atoms, _ = position_batch_center.shape

        # 扩展维度以进行广播
        coords_row = position_batch_center.unsqueeze(2).expand(-1, -1, num_atoms, -1)
        coords_col = position_batch_center.unsqueeze(1).expand(-1, num_atoms, -1, -1)
        # 计算坐标差异
        coord_diff = coords_row - coords_col
        mask = atomic_adjacency_matrix.unsqueeze(-1)  # 扩展维度以匹配
        coord_diff *= mask
        return coord_diff, coords_row, coords_col


    def localframe_features(self, position_batch_center, atomic_adjacency_matrix):
        coord_diff, coord_cross, coord_vertical = self.coordinates2localframe(
            position_batch_center, atomic_adjacency_matrix, norm_diff=True
        )
        batch_size, num_atoms, _, _ = coord_diff.shape
        coord_diff = coord_diff.unsqueeze(-2)
        coord_cross = coord_cross.unsqueeze(-2)
        coord_vertical = coord_vertical.unsqueeze(-2)
        # 由于已经是批处理数据，edges参数不再需要，所有操作都是基于批处理和邻接矩阵
        # 合并局部坐标框架向量
        edge_basis = torch.cat(
            [coord_diff, coord_cross, coord_vertical], dim=-2
        )  # 修改dim参数以正确合并
        # edge_basis.shape
        # r_i 和 r_j 的计算需要调整为适应批处理数据
        # 这里直接使用position_batch_center，因为我们已经有了所有原子对的局部坐标框架向量
        r_i = position_batch_center.unsqueeze(2).expand(
            -1, -1, num_atoms, -1
        )  # [batch_size, num_atoms, num_atoms, 3]
        r_j = position_batch_center.unsqueeze(1).expand(
            -1, num_atoms, -1, -1
        )  # [batch_size, num_atoms, num_atoms, 3]

        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)

        # 计算角度信息
        coff_mul = coff_i * coff_j
        coff_i_norm = torch.norm(coff_i, dim=-1, keepdim=True) + 1e-5
        coff_j_norm = torch.norm(coff_j, dim=-1, keepdim=True) + 1e-5
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm * coff_j_norm)
        pseudo_sin = torch.sqrt(1 - pseudo_cos.pow(2))
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)

        # 合并特征
        coff_feat = torch.cat([pseudo_angle, coff_i, coff_j], dim=-1)
        # coff_feat.shape
        return coff_feat

    def gnnlf2visnet_message_passing_adapter(self, mask, s, atomic_adjacency_matrix, pos):
        # mask: batch_size x atomic_number x atomic_number x hid_dim => 还是类似于领接矩阵表示的边
        # s: batch_size x atomic_number x hid_dim => 类似于节点。
        # - atomic_adjacency_matrix: Smoothed
        # adjacency
        # matrix
        # representing
        # atomic
        # connections.
        # Shape: (batch_size, atomic_number, atomic_number)

        # :param atomic_positions:  the coordinate of each atom in the molecule, shape: batch_size x atomic_number x 3

        lmax = 2
        atomic_number = s.size(1)
        x = s.view(s.size(0) * s.size(1), s.size(2))  # node features
        vec = torch.zeros(
            x.size(0), ((lmax + 1) ** 2) - 1, x.size(1), device=x.device
        )
        # # 生成边索引 edge_index,形状为 (2, edge_number)     => 可以通过生成得到
        # edge_index = torch.randint(0, batch_atomic_number, (2, edge_number))

        batch_idx, rows_idx, cols_idx = torch.nonzero(atomic_adjacency_matrix, as_tuple=True)

        rows = rows_idx + batch_idx * atomic_number
        cols = cols_idx + batch_idx * atomic_number
        edge_index = torch.stack([rows, cols], dim=0)
        # 然后我还要得到edge_weight, 值为adjacent_matrix的值， 形状为(edge_number,)
        edge_weights = atomic_adjacency_matrix[batch_idx, rows_idx, cols_idx]
        # edge_weights = edge_weights.view(-1)
        # 生成边属性 edge_attr,形状为 (edge_number, hidden_dim) => 可以通过生成得到
        edge_attr = mask[batch_idx, rows_idx, cols_idx, :]

        edge_vec = pos[batch_idx, rows_idx, :] - pos[batch_idx, cols_idx, :]

        # 3 => 8
        edge_vec = self.sphere(edge_vec)
        # 生成边向量 edge_vec,形状为 (edge_number, 8) => 可以通过生成得到 看看怎么生成吧
        # edge_vec = torch.randn(edge_number, 8)
        return x, vec, edge_index, edge_weights, edge_attr, edge_vec

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
        batch_size, num_atoms, _ = atomic_positions.shape


        (
            atomic_number_embedding,
            atomic_adjacency_matrix,
            normalized_atom_position_distances,
            edge_features,
        ) = self.mol2graph(atomic_numbers, atomic_positions)
        mask = self.ef_proj(edge_features) * atomic_adjacency_matrix.unsqueeze(-1)
        #  通过一层linear， 再让其特征增加， 并且与领接矩阵相乘， 使得其与领接矩阵的维度一致。
        #  unsqueeze(-1) 表示在最后一个维度上增加一个维度， 这里是增加一个维度， 使得其与mask的维度一致。
        #  mask 的维度为 batch_size x atomic_number x atomic_number x hid_dim
        s = self.neighbor_emb(atomic_numbers, atomic_number_embedding, mask)
        #  s的维度为 batch_size x atomic_number x hid_dim
        v = self.s2v(s, normalized_atom_position_distances, mask)
        #  v的维度为 batch_size x atomic_number x 3 x hid_dim
        if self.global_frame:
            v = torch.sum(v, dim=1, keepdim=True).expand(-1, s.shape[1], -1, -1)
        atomic_direction_feature_list = []

        # TODO: 将colfnet的关于局部坐标框架的特征添投影到hidden_dim的维度上
        # local_frame_featutes = self.localframe_features(position_batch_center=atomic_positions,
        #                                                 atomic_adjacency_matrix=atomic_adjacency_matrix)
        # projected_local_frame_featutes = self.colfnet_features_projection(local_frame_featutes)
        # # #
        # if self.colfnet_features:
        #     atomic_direction_feature_list.append(projected_local_frame_featutes)
        #
        # if self.colfnet_features:
        #     atomic_direction_feature_list.append(local_frame_featutes)
        if self.use_dir1:
            atomic_direction_feature_1 = innerprod(
                v.unsqueeze(1), normalized_atom_position_distances.unsqueeze(-1)
            )
            atomic_direction_feature_list.append(atomic_direction_feature_1)
            # 就是每个元素想乘然后对第三维求
        if self.use_dir2:
            atomic_direction_feature_2 = innerprod(
                v.unsqueeze(2), normalized_atom_position_distances.unsqueeze(-1)
            )
            atomic_direction_feature_list.append(atomic_direction_feature_2)
            # 两者添加的维度位置不同。
        if self.use_dir3:
            atomic_direction_feature_3 = innerprod(
                self.q_proj(v).unsqueeze(1), self.k_proj(v).unsqueeze(2)
            )
            atomic_direction_feature_list.append(atomic_direction_feature_3)
        # dirs 里面的每个元素的维度都是 batch_size x atomic_number x atomic_number x hid_dim

        # batch_size x atomic_number x atomic_number x 8(3+3+2) # 坐标，坐标，角度
        combined_direction_features = torch.cat(
            atomic_direction_feature_list, dim=-1
        )  # batch_size x atomic_number x atomic_number x hid_dim x 2
        # 这个就是把矩阵的最后一个维度进行拼接， 拼接的维度是 hid_dim * (use_dir1 + use_dir2 + use_dir3)
        if self.ev_decay:
            combined_direction_features = (
                combined_direction_features * atomic_adjacency_matrix.unsqueeze(-1)
            )
        if self.add_ef2dir or self.no_filter_decomposition:
            combined_direction_features = torch.cat(
                (combined_direction_features, edge_features), dim=-1
            )  # batch_size x atomic_number x atomic_number x (hid_dim * 2 + ef_dim)
        if self.no_filter_decomposition:
            mask = self.dir_proj(combined_direction_features)
        else:
            dir_project2_max = self.dir_proj(combined_direction_features)
            mask = mask * dir_project2_max
        # 然后就是把dir投影到mask的维度， 然后想乘， 带有广播机制的。
        # 所以这几步的目的就是构建特征矩阵， 把前面的特征全部
        # mask: batch_size x atomic_number x atomic_number x hid_dim => 还是类似于领接矩阵表示的边
        # s: batch_size x atomic_number x hid_dim => 类似于节点。

        if self.use_visnet_message_passing:
            x, vec, edge_index, edge_weight, edge_attr, edge_vec = self.gnnlf2visnet_message_passing_adapter(
                mask, s, atomic_adjacency_matrix, atomic_positions
            )

            # ViS-MP Layers
            for idx, attn in enumerate(self.vis_mp_layers[:-1]):
                dx, dvec, dedge_attr = attn(
                    x, vec, edge_index, edge_weight, edge_attr, edge_vec
                )
                x = x + dx
                vec = vec + dvec
                edge_attr = edge_attr + dedge_attr

            dx, dvec, _ = self.vis_mp_layers[-1](
                x, vec, edge_index, edge_weight, edge_attr, edge_vec
            )
            x = x + dx  # batchsize x atomic_number x hidden_channels
            # batchsize x atomic_number x atomic_number x hidden_channels
            # vec = vec + dvec
            # vec = self.vec_out_norm(vec) # (batch_size * atomic_number) * 8 * hidden_channels

            x = self.out_norm(x)    # (batch_size * atomic_number) * hidden_channels
            x = x.view(batch_size, num_atoms, self.hid_dim)
            s = x

        else:


            for layer_idx, interaction in enumerate(self.interactions):
                if self.no_share_filter:
                    mask = self.ef_projections[layer_idx](
                        edge_features
                    ) * self.dir_projections[layer_idx](combined_direction_features)
                    # mask的形状未改变。
                s = interaction(s, mask) + s
            #     # s的形状未改变。
            # s[atomic_numbers == 0] = 0
            #     d_s, d_mask, _ = interaction(s, mask)
            #     s = s + d_s
            #     mask = mask + d_mask
            # s = self.out_norm(s)    # (batch_size * atomic_number) * hidden_channels

            s[atomic_numbers == 0] = 0


        if self.use_visnet_output_modules:
            pre_reduce_output = self.equivalent_output_module.pre_reduce(
                s, mask, None, None, None
            )
            out = torch.sum(pre_reduce_output, dim=1)
            # out = scatter(x, data.batch, dim=0, reduce=self.reduce_op)
            out = self.equivalent_output_module.post_reduce(out)
        elif self.use_kan_output_modules:
            out = self.kan_output_modules(s)
        else:
            # s: batch_size x atomic_number x hid_dim
            out = self.output_module_1(s) # out : batch_size x atomic_number
            out = out.squeeze(-1)
            out = self.output_module_activation(out)
            out = self.output_module_2(out)   # out : batch_size x 1

        return out
