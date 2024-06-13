import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data

# Define GINConv layer
class GINConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # Use 'add' as the aggregation function
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels + 2, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # Update edge_index and edge_attr to include self loops
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Ensure x_j and edge_attr can be concatenated
        return self.mlp(torch.cat([x_j, edge_attr], dim=-1))

# Define the GINModel class
class GINModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(input_dim, hidden_dim)
        self.conv2 = GINConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x


if __name__ == "__main__":
    # Define graph data
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                               [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
    x = torch.randn(5,
                    3)  # Node features 如果这里的node features 变成 node_number x node_hid_dim0 x node_hid_dim1 x node_hid_dim2
    edge_attr = torch.randn(10,
                            2)  # Edge features        然后edge features变成 edge_number x edge_hid_dim0 x edge_hid_dim1 x edge_hid_dim2

    # Create graph object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Create the GIN model
    model = GINModel(3, 8, 3)  # input_dim, hidden_dim, output_dim

    # Initial node features
    initial_features = data.x

    # Forward pass
    out = model(data.x, data.edge_index, data.edge_attr)

    # Updated node features
    updated_features = out
    print(f"inital features shape: {initial_features.shape}")
    print(f"updated features shape: {updated_features.shape}")