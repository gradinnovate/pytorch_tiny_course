import torch
import torch.optim as optim
from hgr2indices import parse_hgr_file
from hypergraph_model import HypergraphModel
from loss_func import HypergraphRayleighQuotientLoss

def train_hypergraph_model(num_epochs=100, lr=0.01):
    """
    Train hypergraph model on ibm01.hgr dataset with random noise as initial node features.
    """
    # Set device to CPU (MPS has issues with cdist operation in loss function)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load hypergraph data
    print("Loading hypergraph data...")
    hyperedge_index, num_vertices, num_hyperedges = parse_hgr_file("ibm01.hgr")
    hyperedge_index = hyperedge_index.to(device)
    print(f"Loaded graph with {num_vertices} vertices and {num_hyperedges} hyperedges")
    
    # Create model (output_dim=1 to match the loss function expectation)
    model = HypergraphModel(input_dim=1, hidden_dim=16, output_dim=1).to(device)
    
    # Initialize node features with random noise
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(num_vertices, 1).to(device)
    
    # Initialize loss function and optimizer
    criterion = HypergraphRayleighQuotientLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x, hyperedge_index)
        
        # Calculate loss
        loss = criterion(output, hyperedge_index, num_vertices)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    
    print("Training completed!")
    
    # Get final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(x, hyperedge_index)
    
    return model, loss.item(), embeddings, hyperedge_index, num_vertices


def partition_by_median(embeddings):
    """
    Partition nodes into two equal groups based on sorted embedding values.
    
    Args:
        embeddings: Node embeddings tensor of shape (num_nodes, 1)
    
    Returns:
        partition: Binary partition tensor (0 or 1 for each node)
    """
    num_nodes = embeddings.shape[0]
    sorted_indices = torch.argsort(embeddings.squeeze())
    
    # Create equal partitions
    partition = torch.zeros(num_nodes, dtype=torch.long)
    partition[sorted_indices[num_nodes//2:]] = 1
    
    return partition


def calculate_cut_size(partition, hyperedge_index, num_vertices):
    """
    Calculate cut size for hypergraph partitioning.
    
    Args:
        partition: Binary partition tensor (0 or 1 for each node)
        hyperedge_index: Hypergraph edge index
        num_vertices: Number of vertices
    
    Returns:
        cut_size: Number of hyperedges that are cut by the partition
    """
    num_hyperedges = hyperedge_index[1].max() + 1
    cut_size = 0
    
    node_idx, edge_idx = hyperedge_index
    
    for e_idx in range(num_hyperedges):
        # Get nodes in this hyperedge
        nodes_in_edge = node_idx[edge_idx == e_idx]
        
        if len(nodes_in_edge) > 0:
            # Check if hyperedge spans both partitions
            partitions_in_edge = partition[nodes_in_edge]
            unique_partitions = torch.unique(partitions_in_edge)
            
            # If hyperedge contains nodes from both partitions, it's cut
            if len(unique_partitions) > 1:
                cut_size += 1
    
    return cut_size

if __name__ == "__main__":
    trained_model, final_loss, embeddings, hyperedge_index, num_vertices = train_hypergraph_model(num_epochs=500, lr=0.001)
    print(f"Final loss: {final_loss:.6f}")
    
    # Partition based on embedding median
    print("\nPartitioning nodes based on embedding median...")
    partition = partition_by_median(embeddings)
    
    # Calculate cut size
    cut_size = calculate_cut_size(partition, hyperedge_index, num_vertices)
    
    # Print results
    partition_sizes = torch.bincount(partition)
    print(f"Partition sizes: {partition_sizes[0].item()} | {partition_sizes[1].item()}")
    print(f"Cut size: {cut_size}")
    print(f"Total hyperedges: {hyperedge_index[1].max() + 1}")
    print(f"Cut ratio: {cut_size / (hyperedge_index[1].max() + 1):.4f}")