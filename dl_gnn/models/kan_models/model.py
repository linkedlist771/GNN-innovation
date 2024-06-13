from kan import KAN
from torch import nn


class KanOutPutModule(nn.Module):

    def __init__(self, hid_dim: int, atomic_number: int, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.atomic_number = atomic_number
        self.mlp_ouput = nn.Linear(hid_dim, 1)
        self.kan_output = KAN(
            width=[atomic_number, 1], grid=5, k=3, seed=0, device=device
        )
        self.act = nn.SiLU()

    def forward(self, x):
        # input batch_size x atomic_number x hid_dim => 类似于节点。
        # x = x.view(-1, self.hid_dim * self.atomic_number)
        x = self.mlp_ouput(x)
        x = x.squeeze(-1)
        x = self.act(x)
        x = self.kan_output(x)
        return x


def patch_kan_train_function():
    def patch_train(*args, **kwargs):
        pass

    KAN.train = patch_train
