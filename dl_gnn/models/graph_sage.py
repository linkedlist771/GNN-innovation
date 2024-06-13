import os
import numpy as np
from dgl.base import dgl_warning
from dgl.data import AsNodePredDataset, CoraGraphDataset
from torch import nn
import dgl
import torch
from dgl.nn.pytorch.conv.sageconv import SAGEConv


class EarlyStopping:
    def __init__(self, patience: int = -1, checkpoint_path: str = "checkpoint.pth"):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Save model when validation loss decreases."""
        torch.save(model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def close(self):
        os.remove(self.checkpoint_path)


class GraphSAGE(nn.Module):
    def __init__(
        self,
        data_info: dict,
        embed_size: int = -1,
        hidden_size: int = 16,
        num_layers: int = 1,
        activation: str = "relu",
        dropout: float = 0.5,
        aggregator_type: str = "gcn",
    ):
        """GraphSAGE model

        Parameters
        ----------
        data_info : dict
            The information about the input dataset.
        embed_size : int
            The dimension of created embedding table. -1 means using original node embedding
        hidden_size : int
            Hidden size.
        num_layers : int
            Number of hidden layers.
        dropout : float
            Dropout rate.
        activation : str
            Activation function name under torch.nn.functional
        aggregator_type : str
            Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
        """
        super(GraphSAGE, self).__init__()
        self.data_info = data_info
        self.embed_size = embed_size
        if embed_size > 0:
            self.embed = nn.Embedding(data_info["num_nodes"], embed_size)
            in_size = embed_size
        else:
            in_size = data_info["in_size"]
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn.functional, activation)

        for i in range(num_layers):
            in_hidden = hidden_size if i > 0 else in_size
            out_hidden = hidden_size if i < num_layers - 1 else data_info["out_size"]

            self.layers.append(SAGEConv(in_hidden, out_hidden, aggregator_type))

    def forward(self, graph, node_feat, edge_weight=None):
        if self.embed_size > 0:
            dgl_warning(
                "The embedding for node feature is used, and input node_feat is ignored, due to the provided embed_size."
            )
            h = self.embed.weight
        else:
            h = node_feat
        h = self.dropout(h)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h, edge_weight)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h) + h
            # todo, add a read out function
        # 这里graph sage主要是和节点的特征为主，所以这里的readout函数就是对节点特征进行池化
        # 但是在这个情况下， 节点特征并不是很重要，因为都是一样的， 这里比较重要的是
        # 边的特征。
        # 直接进行real out的话好像是没有利用这个h的特征
        # h is 210 \times 1, batch \times 10 \times 1
        h = h.view(-1, 10, 1)  # 10 is batchsize
        h = torch.mean(h, dim=0)
        h_final = torch.mean(h, dim=1, keepdim=True)
        return h_final
        # return dgl.readout_edges(graph, 'feat', op='mean')
        # return h


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def train(cfg, pipeline_cfg, device, data, model, optimizer, loss_fcn):
    g = data[0]  # Only train on the first graph
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    node_feat = g.ndata.get("feat", None)
    edge_feat = g.edata.get("feat", None)
    label = g.ndata["label"]
    train_mask, val_mask, test_mask = (
        g.ndata["train_mask"].bool(),
        g.ndata["val_mask"].bool(),
        g.ndata["test_mask"].bool(),
    )

    stopper = EarlyStopping(**pipeline_cfg["early_stop"])

    val_acc = 0.0
    for epoch in range(pipeline_cfg["num_epochs"]):
        model.train()
        logits = model(g, node_feat, edge_feat)
        loss = loss_fcn(logits[train_mask], label[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], label[train_mask])
        if epoch != 0 and epoch % pipeline_cfg["eval_period"] == 0:
            val_acc = accuracy(logits[val_mask], label[val_mask])

            if stopper.step(val_acc, model):
                break

        print(
            "Epoch {:05d} | Loss {:.4f} | TrainAcc {:.4f} | ValAcc {:.4f}".format(
                epoch, loss.item(), train_acc, val_acc
            )
        )

    stopper.load_checkpoint(model)
    stopper.close()

    model.eval()
    with torch.no_grad():
        logits = model(g, node_feat, edge_feat)
        test_acc = accuracy(logits[test_mask], label[test_mask])
    return test_acc


def train_model_with_early_stopping(
    model, optimizer, loss_fn, dataloader, num_epochs, device
):
    # early_stopping = EarlyStopping(patience=patience, checkpoint_path='checkpoint.pth')
    model = model.to(device)
    model.train()  # 设置模型为训练模式
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    for epoch in range(num_epochs):
        total_loss = 0  # 记录这个epoch的总损失

        for graph_batch, force_batch, energy_batch in dataloader:
            graph_batch = graph_batch.to(device)
            # force_batch = force_batch.to(device)
            energy_batch = energy_batch.to(device).float()
            g = graph_batch  # 获取一个批次的图
            node_feat = g.ndata["feat"].float()  # 提取节点特征
            edge_feat = g.edata["feat"].float()  # 提取边特征
            # node_feat.requires_grad_(True)
            # edge_feat.requires_grad_(True)
            energy_batch.requires_grad_(True)

            pred_val = model(g, node_feat, edge_feat).float()  # 进行预测
            # 计算损失，这里需要你根据实际情况提供目标值
            # 假设我们的目标值存储在 g.ndata['target'] 中
            loss = loss_fn(pred_val, energy_batch)
            # force 暂时不算
            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失
            print(f"loss: {loss.item()}")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")

    #     # 在每个epoch结束时，计算验证集的准确率
    #     model.eval()
    #     val_mse = 0
    #     val_loss = 0
    #     for val_batch in val_dataloader:
    #         val_g = val_batch
    #         val_node_feat = val_g.ndata['feat'].float()
    #         val_edge_feat = val_g.edata['feat'].float()
    #         val_target = val_g.ndata['target'].float()
    #
    #         with torch.no_grad():
    #             val_pred = model(val_g, val_node_feat, val_edge_feat)
    #             val_loss += loss_fn(val_pred, val_target).item()
    #             val_mse += ((val_pred - val_target)**2).mean().item()  # 计算MSE
    #
    #     val_mse /= len(val_dataloader)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}, Val Loss: {val_loss}, Val MSE: {val_mse}")
    #
    #     # 检查是否需要早停
    #     if early_stopping.step(-val_mse, model):  # 注意这里我们传入-val_mse，因为我们希望MSE尽可能小
    #         print("Early stopping")
    #         break
    #
    # # 加载最好的模型参数
    # early_stopping.load_checkpoint(model)
    # early_stopping.close()


def main(run, cfg, data):
    device = cfg["device"]
    pipeline_cfg = cfg["general_pipeline"]
    # create model
    model = GraphSAGE(**cfg["model"])
    model = model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **pipeline_cfg["optimizer"])
    # train
    test_acc = train(cfg, pipeline_cfg, device, data, model, optimizer, loss)
    torch.save(
        {"cfg": cfg, "model": model.state_dict()},
        os.path.join(pipeline_cfg["save_path"], "run_{}.pth".format(run)),
    )

    return test_acc


if __name__ == "__main__":

    # load data
    data = AsNodePredDataset(CoraGraphDataset())
    data_zero = data[0]
    # cfg if load from a yaml file
    import yaml

    with open("../configs/nodepred_cora_sage.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model_cfg = cfg["model"]
    cfg["model"]["data_info"] = {
        "in_size": (
            model_cfg["embed_size"]
            if model_cfg["embed_size"] > 0
            else data[0].ndata["feat"].shape[1]
        ),
        "out_size": data.num_classes,
        "num_nodes": data[0].num_nodes(),
    }
    if not os.path.exists(cfg["general_pipeline"]["save_path"]):
        os.makedirs(cfg["general_pipeline"]["save_path"])

    all_acc = []
    num_runs = 1
    for run in range(num_runs):
        print(f"Run experiment #{run}")
        test_acc = main(run, cfg, data)
        print("Test Accuracy {:.4f}".format(test_acc))
        all_acc.append(test_acc)
    avg_acc = np.round(np.mean(all_acc), 6)
    std_acc = np.round(np.std(all_acc), 6)
    print(f"Accuracy across {num_runs} runs: {avg_acc} ± {std_acc}")
