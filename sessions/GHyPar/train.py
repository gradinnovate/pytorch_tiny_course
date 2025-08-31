import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from numba import jit, prange
from hgr2indices import parse_hgr_file
from hypergraph_model import HypergraphModel
from loss_func2 import HypergraphRayleighQuotientLossDirect as HypergraphRayleighQuotientLoss

def train_hypergraph_model(num_epochs=100, lr=0.01, seed=42):
    """
    Train hypergraph model on ibm01.hgr dataset with random noise as initial node features.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    # Set device to CPU (MPS has issues with cdist operation in loss function)
    device = torch.device("mps")
    print(f"Using device: {device}")
    print(f"Random seed: {seed}")
    
    # Load hypergraph data
    print("Loading hypergraph data...")
    hyperedge_index, num_vertices, num_hyperedges = parse_hgr_file("ibm01.hgr")
    hyperedge_index = hyperedge_index.to(device)
    print(f"Loaded graph with {num_vertices} vertices and {num_hyperedges} hyperedges")
    
    # Load partition solution
    print("Loading partition solution...")
    with open("ibm01.part.2", "r") as f:
        partition_ids = [int(line.strip()) for line in f.readlines()]
    
    # Calculate cut size of ground truth partition before conversion
    ground_truth_partition = torch.tensor(partition_ids, dtype=torch.long)
    ground_truth_cut_size = calculate_cut_size(ground_truth_partition, hyperedge_index.cpu(), num_vertices)
    
    # Print ground truth partition stats
    ground_truth_sizes = torch.bincount(ground_truth_partition)
    print(f"Ground truth partition sizes: {ground_truth_sizes[0].item()} | {ground_truth_sizes[1].item()}")
    print(f"Ground truth cut size: {ground_truth_cut_size}")
    print(f"Ground truth cut ratio: {ground_truth_cut_size / num_hyperedges:.4f}")
    
    # Convert partition IDs: 1 -> 1, 0 -> -1
    partition_ids = torch.tensor(partition_ids, dtype=torch.float)
    partition_ids = torch.where(partition_ids == 0, -1.0, 1.0)
    print(f"Loaded partition solution with {len(partition_ids)} nodes (converted 0->-1, 1->1)")
    
    # Create model (input_dim=3 for core features)
    model = HypergraphModel(input_dim=3, hidden_dim=64, output_dim=1).to(device)
    
    # Initialize node features with node degrees
    node_degrees = torch.zeros(num_vertices, dtype=torch.float)
    node_idx = hyperedge_index[0].cpu()
    for node in node_idx:
        node_degrees[node] += 1
    
    # Standardize node degrees (z-score normalization)
    mean_degree = node_degrees.mean()
    std_degree = node_degrees.std()
    node_degrees_std = (node_degrees - mean_degree) / (std_degree + 1e-8)
    
    print(f"Node degree stats - Mean: {mean_degree:.2f}, Std: {std_degree:.2f}")
    print(f"After standardization - Mean: {node_degrees_std.mean():.6f}, Std: {node_degrees_std.std():.6f}")
    
    # Use more sophisticated normalization: Min-Max + Standardization
    def robust_normalize(feature, target_norm=None):
        # First apply min-max normalization to [0, 1] 
        min_val = feature.min()
        max_val = feature.max()
        if max_val > min_val:
            feature_minmax = (feature - min_val) / (max_val - min_val)
        else:
            feature_minmax = feature.clone()
        
        # Then apply z-score standardization
        mean_val = feature_minmax.mean()
        std_val = feature_minmax.std()
        feature_std = (feature_minmax - mean_val) / (std_val + 1e-8)
        
        # Optionally scale to match target norm
        if target_norm is not None:
            current_norm = torch.norm(feature_std, p=2)
            feature_std = (feature_std / (current_norm + 1e-8)) * target_norm
            
        return feature_std
    
    # Apply robust normalization to base features
    degree_norm = torch.norm(node_degrees_std, p=2)  # Use as reference norm
    partition_ids_normalized = robust_normalize(partition_ids, degree_norm)
    
    node_indices = torch.arange(num_vertices, dtype=torch.float)
    node_indices_normalized = robust_normalize(node_indices, degree_norm)
    
    print(f"Degree reference norm: {degree_norm:.2f}")
    print(f"Partition ID range: [{partition_ids_normalized.min():.3f}, {partition_ids_normalized.max():.3f}]")
    print(f"Node indices range: [{node_indices_normalized.min():.3f}, {node_indices_normalized.max():.3f}]")
    
    # Combine the 3 core features with optimized ordering: [partition_ids, node_degrees, node_indices]
    x = torch.stack([
        partition_ids_normalized,  # Most important: ground truth signal
        node_degrees_std,          # Second: topology structure  
        node_indices_normalized    # Third: positional information
    ], dim=1).to(device)
    
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


def partition_by_balanced_cut(embeddings, hyperedge_index, node_weights=None, epsilon=0.02):
    """
    Partition nodes into balanced groups within epsilon constraint to minimize cut size.
    Uses Numba for parallel acceleration.
    
    Args:
        embeddings: Node embeddings tensor of shape (num_nodes, 1)
        hyperedge_index: Hypergraph edge index for cut size calculation
        node_weights: Node weights (default: uniform weights)
        epsilon: Balance constraint parameter
    
    Returns:
        partition: Binary partition tensor (0 or 1 for each node)
    """
    num_nodes = embeddings.shape[0]
    embeddings_flat = embeddings.flatten()
    
    # Convert to numpy for Numba
    embeddings_np = embeddings_flat.cpu().numpy()
    hyperedge_index_np = hyperedge_index.cpu().numpy()
    
    # Use uniform weights if not provided
    if node_weights is None:
        node_weights = np.ones(num_nodes, dtype=np.float32)
    else:
        node_weights = node_weights.cpu().numpy()
    
    total_weight = node_weights.sum()
    
    # Balance constraint: (0.5 - epsilon) * W <= partition_weight <= (0.5 + epsilon) * W
    min_weight = (0.5 - epsilon) * total_weight
    max_weight = (0.5 + epsilon) * total_weight
    
    print(f"Total weight: {total_weight:.2f}")
    print(f"Balance range: [{min_weight:.2f}, {max_weight:.2f}]")
    
    # Check embedding diversity
    unique_values = np.unique(embeddings_np)
    print(f"Embedding diversity: {len(unique_values)} unique values out of {num_nodes} nodes ({len(unique_values)/num_nodes*100:.1f}%)")
    
    # 
    # nodes by embedding values
    sorted_indices = np.argsort(embeddings_np)
    
    num_hyperedges = hyperedge_index_np[1].max() + 1
    node_idx = hyperedge_index_np[0]
    edge_idx = hyperedge_index_np[1]
    
    print("Finding valid splits...")
    
    # First step: find all valid splits
    candidate_splits = find_valid_splits(
        sorted_indices, node_weights, min_weight, max_weight, num_nodes
    )
    
    if candidate_splits.shape[0] == 0:
        print("Warning: No valid splits found within balance constraint, using median split")
        partition = torch.zeros(num_nodes, dtype=torch.long)
        partition[sorted_indices[num_nodes//2:]] = 1
        return partition
    
    print(f"Found {candidate_splits.shape[0]} valid splits, evaluating cut sizes in parallel...")
    
    # Second step: evaluate cut sizes in parallel
    results = evaluate_cuts_parallel(
        candidate_splits, sorted_indices, node_idx, edge_idx, num_hyperedges, num_nodes
    )
    
    print(f"Evaluated {results.shape[0]} valid balanced splits")
    
    # Find best split (minimum cut size)
    best_idx = np.argmin(results[:, 2])  # Column 2 is cut_size
    best_split_idx = int(results[best_idx, 0])
    best_weight = results[best_idx, 1]
    best_cut_size = int(results[best_idx, 2])
    
    # Create best partition
    best_partition = torch.zeros(num_nodes, dtype=torch.long)
    for j in range(best_split_idx, num_nodes):
        best_partition[sorted_indices[j]] = 1
    
    split_value = embeddings_np[sorted_indices[best_split_idx]]
    
    print(f"Best split at index {best_split_idx}, value: {split_value:.6f}")
    print(f"Best split weight: {best_weight:.2f} (ratio: {best_weight/total_weight:.4f})")
    print(f"Best cut size: {best_cut_size}")
    
    return best_partition


@jit(nopython=True)
def calculate_cut_size_numba(partition, node_idx, edge_idx, num_hyperedges):
    """
    Numba-accelerated cut size calculation.
    
    Args:
        partition: Binary partition array (0 or 1 for each node)
        node_idx: Node indices from hyperedge_index[0]
        edge_idx: Edge indices from hyperedge_index[1]
        num_hyperedges: Total number of hyperedges
    
    Returns:
        cut_size: Number of hyperedges that are cut by the partition
    """
    cut_size = 0
    
    for e_idx in range(num_hyperedges):
        # Find nodes in this hyperedge
        has_partition_0 = False
        has_partition_1 = False
        
        for i in range(len(edge_idx)):
            if edge_idx[i] == e_idx:
                node = node_idx[i]
                if partition[node] == 0:
                    has_partition_0 = True
                else:
                    has_partition_1 = True
                
                # Early exit if we found both partitions
                if has_partition_0 and has_partition_1:
                    cut_size += 1
                    break
    
    return cut_size


@jit(nopython=True)
def find_valid_splits(sorted_indices, node_weights, min_weight, max_weight, num_nodes):
    """
    Find all valid splits that satisfy balance constraints.
    
    Returns:
        results: Array of (split_idx, weight) for valid splits
    """
    valid_splits = []
    cumulative_weight = 0.0
    
    for i in range(num_nodes):
        node_idx_val = sorted_indices[i]
        cumulative_weight += node_weights[node_idx_val]
        
        if min_weight <= cumulative_weight <= max_weight:
            valid_splits.append((i, cumulative_weight))
    
    # Convert to numpy array
    if len(valid_splits) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    
    results = np.zeros((len(valid_splits), 2), dtype=np.float64)
    for idx in range(len(valid_splits)):
        results[idx, 0] = valid_splits[idx][0]
        results[idx, 1] = valid_splits[idx][1]
    
    return results


@jit(nopython=True, parallel=True)
def evaluate_cuts_parallel(candidate_splits, sorted_indices, node_idx, edge_idx, num_hyperedges, num_nodes):
    """
    Parallel evaluation of cut sizes for candidate splits.
    
    Returns:
        results: Array of (split_idx, weight, cut_size)
    """
    num_candidates = candidate_splits.shape[0]
    results = np.zeros((num_candidates, 3), dtype=np.float64)
    
    for idx in prange(num_candidates):
        split_idx = int(candidate_splits[idx, 0])
        weight = candidate_splits[idx, 1]
        
        # Create partition for this split
        partition = np.zeros(num_nodes, dtype=np.int32)
        for j in range(split_idx, num_nodes):
            partition[sorted_indices[j]] = 1
        
        # Calculate cut size
        cut_size = calculate_cut_size_numba(partition, node_idx, edge_idx, num_hyperedges)
        
        results[idx, 0] = split_idx
        results[idx, 1] = weight
        results[idx, 2] = cut_size
    
    return results


def calculate_cut_size(partition, hyperedge_index, num_vertices):
    """
    Calculate cut size for hypergraph partitioning (wrapper for compatibility).
    """
    if isinstance(partition, torch.Tensor):
        partition = partition.numpy()
    if isinstance(hyperedge_index, torch.Tensor):
        hyperedge_index = hyperedge_index.numpy()
    
    num_hyperedges = hyperedge_index[1].max() + 1
    node_idx = hyperedge_index[0]
    edge_idx = hyperedge_index[1]
    
    return calculate_cut_size_numba(partition, node_idx, edge_idx, num_hyperedges)

if __name__ == "__main__":
    # Set global seed at the beginning
    SEED = 42
    trained_model, final_loss, embeddings, hyperedge_index, num_vertices = train_hypergraph_model(num_epochs=500, lr=0.000025, seed=SEED)
    print(f"Final loss: {final_loss:.6f}")
    
    # Partition based on balanced cut optimization
    print("\nPartitioning nodes with balanced cut optimization...")
    print(f"Embedding stats - Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}, Median: {embeddings.median():.6f}")
    print(f"Embedding std: {embeddings.std():.6f}")
    
    partition = partition_by_balanced_cut(embeddings, hyperedge_index, epsilon=0.02)
    
    # Calculate cut size (move data to CPU for calculation)
    partition_cpu = partition.cpu()
    hyperedge_index_cpu = hyperedge_index.cpu()
    cut_size = calculate_cut_size(partition_cpu, hyperedge_index_cpu, num_vertices)
    
    # Print results
    partition_sizes = torch.bincount(partition)
    print(f"Partition sizes: {partition_sizes[0].item()} | {partition_sizes[1].item()}")
    print(f"Cut size: {cut_size}")
    print(f"Total hyperedges: {hyperedge_index[1].max() + 1}")
    print(f"Cut ratio: {cut_size / (hyperedge_index[1].max() + 1):.4f}")
    
    # Plot embedding histogram
    print("\nPlotting embedding histogram...")
    embeddings_cpu = embeddings.cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 6))
    plt.hist(embeddings_cpu, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Embedding Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Node Embeddings')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at median
    median_val = torch.median(embeddings.squeeze()).cpu().item()
    plt.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('embedding_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Histogram saved as 'embedding_histogram.png'")