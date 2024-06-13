from dgl.dataloading import GraphDataLoader
from dl_gnn.models.graph_sage import GraphSAGE, train_model_with_early_stopping
from dl_gnn.data.md17 import GraphSageMD17Dataset

# import mse criteria from torch
from torch.nn import MSELoss
from torch.optim import Adam
import yaml

with open("../configs/nodepred_cora_sage.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

md17_aspirin_path = r"C:\Users\23174\Desktop\GitHub Project\GitHubProjectBigData\GNN-Molecular-Project\MD17\raw\md17_aspirin.npz"
graph_sage_md17_dataset = GraphSageMD17Dataset(raw_data_or_data_path=md17_aspirin_path)
graph_sage_md17_dataset.build_graphs()
graph_sage_md17_dataloader = GraphDataLoader(
    graph_sage_md17_dataset, batch_size=10, shuffle=True
)

# 获取第一个批次的数据
for graph_batch, force_batch, energy_batch in graph_sage_md17_dataloader:
    # batch_data 将包含一个批次的图以及任何其他信息（如标签等）
    g = graph_batch
    break  # 我们只需要第一个批次的数据

model_cfg = cfg["model"]

cfg["model"]["data_info"] = {
    "in_size": (
        model_cfg["embed_size"]
        if model_cfg["embed_size"] > 0
        else g.ndata["feat"].shape[1]
    ),
    "out_size": 1,
    "num_nodes": g.num_nodes(),
}
model = GraphSAGE(**cfg["model"])


for name, param in model.named_parameters():
    param.requires_grad_(True)

num_epochs = 1000
optimizer = Adam(model.parameters(), lr=1e-1)
criteria = MSELoss()
train_model_with_early_stopping(
    model=model,
    num_epochs=num_epochs,
    optimizer=optimizer,
    loss_fn=criteria,
    dataloader=graph_sage_md17_dataloader,
    device="cuda:0",
)
# # 提取节点特征和边特征
# node_feat = g.ndata['feat']  # 假设节点特征存储在 'feat' 键下
# edge_feat = g.edata['feat']  # 假设边特征存储在 'feat' 键下
#
# # 打印特征形状
# print(f"node_feature shape: {node_feat.shape}")
# print(f"edge_feature shape: {edge_feat.shape}")
#
# # 将模型置于训练模式
# model.train()
#
# # 进行预测
# pred_val = model(g, node_feat.float(), edge_feat.float())
#
# # 打印输出
# print(f"logits.shape: {pred_val.shape}")
# print(f'pred_val: {pred_val}')
