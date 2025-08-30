import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, BatchNorm

class HypergraphModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super(HypergraphModel, self).__init__()
        
        self.conv1 = HypergraphConv(input_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim)
        self.conv3 = HypergraphConv(hidden_dim, hidden_dim)
        self.conv4 = HypergraphConv(hidden_dim, output_dim)
        self.bn = BatchNorm(hidden_dim)
        
        # Xavier initialization
        self._init_weights()
        
    def forward(self, x, hyperedge_index):
        # First layer
        x = self.conv1(x, hyperedge_index)
        x = self.bn(x)
        x = F.relu(x)
        
        # Second layer
        x = self.conv2(x, hyperedge_index)
        x = F.relu(x)
        
        # Third layer
        x = self.conv3(x, hyperedge_index)
        x = F.relu(x)
        
        x = self.conv4(x, hyperedge_index)
        
        return x
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if len(module.weight.shape) > 1:  # Only for 2D+ tensors
                    nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    from hgr2indices import parse_hgr_file
    
    # Load hypergraph data
    hyperedge_index, num_vertices, num_hyperedges = parse_hgr_file("ibm01.hgr")
    
    # Create model
    model = HypergraphModel(input_dim=1, hidden_dim=32, output_dim=32)
    
    # Create dummy node features (all ones)
    x = torch.ones(num_vertices, 1)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    print(f"Hyperedge index shape: {hyperedge_index.shape}")
    
    output = model(x, hyperedge_index)
    print(f"Output shape: {output.shape}")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")