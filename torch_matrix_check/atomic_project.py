import torch
import torch.nn as nn

# 假设的输入张量维度
batch_size = 4
atomic_number = 10
hid_dim = 128

# 输入数据
input_tensor = torch.randn(batch_size, atomic_number, hid_dim)

# 定义网络层
output_module_1 = nn.Linear(hid_dim, 1)  # 第一个线性层
output_module_activation = nn.SiLU(inplace=True)  # 激活函数
output_module_2 = nn.Linear(atomic_number, 1)  # 第二个线性层

# 将输入张量压缩为二维，以适应线性层的要求

# 应用网络层
output = output_module_1(input_tensor)
# 去掉最后一个维度为1的维度
output = output.squeeze(-1)

output = output_module_activation(output)
output = output_module_2(output)


print(output.shape)  # 输出张量的形状应该是 (batch_size, 1)
