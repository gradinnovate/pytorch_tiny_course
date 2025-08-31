"""
Hypergraph Partitioning with Neural Networks

This module implements a hypergraph partitioning system using GNNs with spectral loss functions.
It includes feature engineering, balanced partitioning, and Numba-accelerated optimization.
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Tuple, Optional, Dict, Any, List
from numba import jit, prange, set_num_threads
import os

# Set Numba to use more threads for parallel sections
num_threads = min(8, os.cpu_count())  # Use up to 8 threads
set_num_threads(num_threads)

from hgr2indices import parse_hgr_file
from hypergraph_model import HypergraphModel
from loss_func2 import HypergraphRayleighQuotientLossDirect as HypergraphRayleighQuotientLoss

# Constants
DEFAULT_SEED = 42
DEFAULT_EPSILON = 0.02
DEFAULT_NUM_EPOCHS = 1500  # Back to original
DEFAULT_LR = 0.00002  # Back to original working LR
DEFAULT_HIDDEN_DIM = 32

def set_random_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_hypergraph_data(file_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load hypergraph data from .hgr file including hyperedge weights."""
    print("Loading hypergraph data...")
    hyperedge_index, hyperedge_weights, num_vertices, num_hyperedges = parse_hgr_file(file_path)
    hyperedge_index = hyperedge_index.to(device)
    hyperedge_weights = hyperedge_weights.to(device)
    print(f"Loaded graph with {num_vertices} vertices and {num_hyperedges} hyperedges")
    print(f"Hyperedge weight stats - Mean: {hyperedge_weights.mean():.2f}, "
          f"Min: {hyperedge_weights.min():.0f}, Max: {hyperedge_weights.max():.0f}")
    return hyperedge_index, hyperedge_weights, num_vertices, num_hyperedges


def load_partition_solution(file_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load ground truth partition solution and compute statistics."""
    print(f"Loading partition solution from {file_path}...")
    with open(file_path, "r") as f:
        partition_ids = [int(line.strip()) for line in f.readlines()]
    
    # Keep original for cut size calculation
    ground_truth_partition = torch.tensor(partition_ids, dtype=torch.long)
    
    # Convert to bipolar encoding: 0 -> -1, 1 -> 1
    partition_ids = torch.tensor(partition_ids, dtype=torch.float)
    partition_ids = torch.where(partition_ids == 0, -1.0, 1.0)
    
    stats = {
        'ground_truth_partition': ground_truth_partition,
        'partition_ids': partition_ids,
        'num_nodes': len(partition_ids)
    }
    
    print(f"Loaded partition solution with {len(partition_ids)} nodes (converted 0->-1, 1->1)")
    return partition_ids, stats


def load_multiple_partition_solutions(hints_dir: str = "hints") -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Load multiple partition solutions from hints directory."""
    import glob
    
    # Find all hint files
    hint_files = glob.glob(os.path.join(hints_dir, "ibm01.hgr.k.2.*"))
    hint_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(hint_files)} partition solutions in {hints_dir}/")
    
    solutions = []
    for i, file_path in enumerate(hint_files):
        print(f"Loading solution {i+1}/{len(hint_files)}: {os.path.basename(file_path)}")
        partition_ids, stats = load_partition_solution(file_path)
        solutions.append((partition_ids, stats))
    
    return solutions


def compute_ground_truth_stats(ground_truth_partition: torch.Tensor, 
                              hyperedge_index: torch.Tensor, 
                              num_hyperedges: int) -> None:
    """Compute and print ground truth partition statistics."""
    ground_truth_cut_size = calculate_cut_size(ground_truth_partition, hyperedge_index.cpu(), 0)
    ground_truth_sizes = torch.bincount(ground_truth_partition)
    
    print(f"Ground truth partition sizes: {ground_truth_sizes[0].item()} | {ground_truth_sizes[1].item()}")
    print(f"Ground truth cut size: {ground_truth_cut_size}")
    print(f"Ground truth cut ratio: {ground_truth_cut_size / num_hyperedges:.4f}")


def compute_node_degrees(hyperedge_index: torch.Tensor, num_vertices: int) -> torch.Tensor:
    """Compute node degrees from hypergraph structure."""
    node_degrees = torch.zeros(num_vertices, dtype=torch.float)
    node_idx = hyperedge_index[0].cpu()
    for node in node_idx:
        node_degrees[node] += 1
    return node_degrees




def robust_normalize(feature: torch.Tensor, target_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply robust normalization: Min-Max followed by Z-score standardization.
    
    Args:
        feature: Input feature tensor
        target_norm: Optional target norm to scale the result
        
    Returns:
        Normalized feature tensor
    """
    # Min-max normalization to [0, 1]
    min_val = feature.min()
    max_val = feature.max()
    if max_val > min_val:
        feature_minmax = (feature - min_val) / (max_val - min_val)
    else:
        feature_minmax = feature.clone()
    
    # Z-score standardization
    mean_val = feature_minmax.mean()
    std_val = feature_minmax.std()
    feature_std = (feature_minmax - mean_val) / (std_val + 1e-8)
    
    # Scale to target norm if provided
    if target_norm is not None:
        current_norm = torch.norm(feature_std, p=2)
        feature_std = (feature_std / (current_norm + 1e-8)) * target_norm
        
    return feature_std


def create_node_features(partition_ids: torch.Tensor, 
                        hyperedge_index: torch.Tensor, 
                        num_vertices: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Create and normalize node features for the hypergraph model.
    
    Returns:
        Tuple of (feature_tensor, stats_dict)
    """
    # Compute node degrees
    node_degrees = compute_node_degrees(hyperedge_index, num_vertices)
    
    # Standardize node degrees (z-score)
    mean_degree = node_degrees.mean()
    std_degree = node_degrees.std()
    node_degrees_std = (node_degrees - mean_degree) / (std_degree + 1e-8)
    
    # Use degree norm as reference for other features
    degree_norm = torch.norm(node_degrees_std, p=2)
    
    # Robust normalization for partition IDs and node indices
    partition_ids_normalized = robust_normalize(partition_ids, degree_norm)
    
    node_indices = torch.arange(num_vertices, dtype=torch.float)
    node_indices_normalized = robust_normalize(node_indices, degree_norm)
    
    # Combine features: [partition_ids, node_degrees, node_indices]
    features = torch.stack([
        partition_ids_normalized,  # Most important: ground truth signal
        node_degrees_std,          # Second: topology structure  
        node_indices_normalized    # Third: positional information
    ], dim=1)
    
    # Statistics for logging
    stats = {
        'mean_degree': mean_degree.item(),
        'std_degree': std_degree.item(),
        'degree_norm': degree_norm.item(),
        'partition_range': [partition_ids_normalized.min().item(), partition_ids_normalized.max().item()],
        'indices_range': [node_indices_normalized.min().item(), node_indices_normalized.max().item()]
    }
    
    return features, stats


def train_model_multi_sample(model: HypergraphModel, 
                           features_list: List[torch.Tensor], 
                           hyperedge_index: torch.Tensor,
                           hyperedge_weights: torch.Tensor,
                           num_vertices: int,
                           num_epochs: int = DEFAULT_NUM_EPOCHS,
                           lr: float = DEFAULT_LR) -> Tuple[torch.Tensor, float]:
    """
    Train the hypergraph model with multiple partition solution samples.
    
    Args:
        features_list: List of feature tensors for different partition solutions
    
    Returns:
        Tuple of (final_embeddings, final_loss)
    """
    criterion = HypergraphRayleighQuotientLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    num_samples = len(features_list)
    print(f"Starting training with {num_samples} different partition solutions...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Train on all samples in each epoch
        for sample_idx, features in enumerate(features_list):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(features, hyperedge_index)
            
            # Calculate loss
            loss = criterion(output, hyperedge_index, num_vertices)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss across all samples
        avg_loss = epoch_loss / num_samples
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.6f}")
    
    print("Training completed!")
    
    # Get final embeddings using evaluation features (ibm01.part.2)
    # This will be passed from the main function
    model.eval()
    return model, avg_loss  # Return model instead of embeddings


def train_hypergraph_model(num_epochs: int = DEFAULT_NUM_EPOCHS, 
                          lr: float = DEFAULT_LR, 
                          seed: int = DEFAULT_SEED) -> Tuple[HypergraphModel, float, torch.Tensor, torch.Tensor, int]:
    """
    Main training function for hypergraph partitioning model.
    
    Args:
        num_epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, final_loss, embeddings, hyperedge_index, num_vertices)
    """
    # Setup
    set_random_seeds(seed)
    device = torch.device("mps")
    print(f"Using device: {device}")
    print(f"Random seed: {seed}")
    
    # Load data with hyperedge weights
    hyperedge_index, hyperedge_weights, num_vertices, num_hyperedges = load_hypergraph_data("ibm01.hgr", device)
    
    # Load multiple partition solutions from hints directory for training
    solutions = load_multiple_partition_solutions("hints")
    
    # Create features for all partition solutions
    features_list = []
    for i, (partition_ids, partition_stats) in enumerate(solutions):
        print(f"\nProcessing solution {i+1}/{len(solutions)}...")
        features, feature_stats = create_node_features(partition_ids, hyperedge_index, num_vertices)
        features = features.to(device)
        features_list.append(features)
    
    # Print feature statistics from the last processed solution
    print(f"\nFeature statistics (from last solution):")
    print(f"Node degree stats - Mean: {feature_stats['mean_degree']:.2f}, Std: {feature_stats['std_degree']:.2f}")
    print(f"Degree reference norm: {feature_stats['degree_norm']:.2f}")
    print(f"Partition ID range: [{feature_stats['partition_range'][0]:.3f}, {feature_stats['partition_range'][1]:.3f}]")
    print(f"Node indices range: [{feature_stats['indices_range'][0]:.3f}, {feature_stats['indices_range'][1]:.3f}]")
    
    # Create and train model with multiple training samples from hints
    model = HypergraphModel(input_dim=3, hidden_dim=DEFAULT_HIDDEN_DIM, output_dim=1).to(device)
    trained_model, final_loss = train_model_multi_sample(model, features_list, hyperedge_index, hyperedge_weights, num_vertices, num_epochs, lr)
    
    # Load evaluation data (ibm01.part.2)
    print("\n" + "="*50)
    print("Loading evaluation data (ibm01.part.2)...")
    eval_partition_ids, eval_stats = load_partition_solution("ibm01.part.2")
    
    # Compute ground truth statistics only for evaluation data
    compute_ground_truth_stats(eval_stats['ground_truth_partition'], hyperedge_index, num_hyperedges)
    
    eval_features, eval_feature_stats = create_node_features(eval_partition_ids, hyperedge_index, num_vertices)
    eval_features = eval_features.to(device)
    
    # Get embeddings using evaluation features
    trained_model.eval()
    with torch.no_grad():
        embeddings = trained_model(eval_features, hyperedge_index)
    
    print(f"Generated embeddings using evaluation features (ibm01.part.2)")
    
    return trained_model, final_loss, embeddings, hyperedge_index, num_vertices


# Numba-accelerated functions for balanced partitioning
@jit(nopython=True)
def calculate_cut_size_numba_fast(partition: np.ndarray, 
                                node_idx: np.ndarray, 
                                edge_idx: np.ndarray, 
                                num_hyperedges: int) -> int:
    """
    Optimized Numba-accelerated cut size calculation.
    Pre-compute hyperedge partitions for faster evaluation.
    """
    cut_size = 0
    
    # Pre-allocate arrays for hyperedge partition counts
    partition_0_count = np.zeros(num_hyperedges, dtype=np.int32)
    partition_1_count = np.zeros(num_hyperedges, dtype=np.int32)
    
    # Count partition memberships for each hyperedge
    for i in range(len(node_idx)):
        node = node_idx[i]
        edge = edge_idx[i]
        
        if partition[node] == 0:
            partition_0_count[edge] += 1
        else:
            partition_1_count[edge] += 1
    
    # Count cut hyperedges (those with nodes in both partitions)
    for e_idx in range(num_hyperedges):
        if partition_0_count[e_idx] > 0 and partition_1_count[e_idx] > 0:
            cut_size += 1
    
    return cut_size


@jit(nopython=True)
def find_valid_splits(sorted_indices: np.ndarray, 
                     node_weights: np.ndarray, 
                     min_weight: float, 
                     max_weight: float, 
                     num_nodes: int) -> np.ndarray:
    """
    Find all valid splits that satisfy balance constraints.
    
    Returns:
        Array of (split_idx, weight) for valid splits
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
def evaluate_cuts_parallel(candidate_splits: np.ndarray, 
                          sorted_indices: np.ndarray, 
                          node_idx: np.ndarray, 
                          edge_idx: np.ndarray, 
                          num_hyperedges: int, 
                          num_nodes: int) -> np.ndarray:
    """
    Parallel evaluation of cut sizes for candidate splits.
    
    Returns:
        Array of (split_idx, weight, cut_size)
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
        
        # Calculate cut size using optimized version
        cut_size = calculate_cut_size_numba_fast(partition, node_idx, edge_idx, num_hyperedges)
        
        results[idx, 0] = split_idx
        results[idx, 1] = weight
        results[idx, 2] = cut_size
    
    return results


def partition_by_balanced_cut(embeddings: torch.Tensor, 
                             hyperedge_index: torch.Tensor, 
                             node_weights: Optional[np.ndarray] = None, 
                             epsilon: float = DEFAULT_EPSILON) -> torch.Tensor:
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
    
    # Sort nodes by embedding values
    sorted_indices = np.argsort(embeddings_np)
    
    num_hyperedges = hyperedge_index_np[1].max() + 1
    node_idx = hyperedge_index_np[0]
    edge_idx = hyperedge_index_np[1]
    
    print("Finding valid splits...")
    
    # First step: find all valid splits
    candidate_splits = find_valid_splits(sorted_indices, node_weights, min_weight, max_weight, num_nodes)
    
    if candidate_splits.shape[0] == 0:
        print("Warning: No valid splits found within balance constraint, using median split")
        partition = torch.zeros(num_nodes, dtype=torch.long)
        partition[sorted_indices[num_nodes//2:]] = 1
        return partition
    
    print(f"Found {candidate_splits.shape[0]} valid splits, evaluating cut sizes in parallel with {num_threads} threads...")
    
    # Second step: evaluate cut sizes in parallel
    import time
    start_time = time.time()
    results = evaluate_cuts_parallel(candidate_splits, sorted_indices, node_idx, edge_idx, num_hyperedges, num_nodes)
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    
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


def calculate_cut_size(partition: torch.Tensor, 
                      hyperedge_index: torch.Tensor, 
                      num_vertices: int) -> int:
    """Calculate cut size for hypergraph partitioning (wrapper for compatibility)."""
    if isinstance(partition, torch.Tensor):
        partition = partition.numpy()
    if isinstance(hyperedge_index, torch.Tensor):
        hyperedge_index = hyperedge_index.numpy()
    
    num_hyperedges = hyperedge_index[1].max() + 1
    node_idx = hyperedge_index[0]
    edge_idx = hyperedge_index[1]
    
    return calculate_cut_size_numba_fast(partition, node_idx, edge_idx, num_hyperedges)


def plot_embedding_histogram(embeddings: torch.Tensor, save_path: str = 'embedding_histogram.png') -> None:
    """Plot and save embedding distribution histogram."""
    print("\\nPlotting embedding histogram...")
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Histogram saved as '{save_path}'")


def main() -> None:
    """Main execution function."""
    # Train model
    trained_model, final_loss, embeddings, hyperedge_index, num_vertices = train_hypergraph_model(
        num_epochs=DEFAULT_NUM_EPOCHS, 
        lr=DEFAULT_LR, 
        seed=DEFAULT_SEED
    )
    print(f"Final loss: {final_loss:.6f}")
    
    # Analyze embeddings
    print("\\nPartitioning nodes with balanced cut optimization...")
    print(f"Embedding stats - Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}, Median: {embeddings.median():.6f}")
    print(f"Embedding std: {embeddings.std():.6f}")
    
    # Partition with balanced cut
    partition = partition_by_balanced_cut(embeddings, hyperedge_index, epsilon=DEFAULT_EPSILON)
    
    # Calculate and report results
    partition_cpu = partition.cpu()
    hyperedge_index_cpu = hyperedge_index.cpu()
    cut_size = calculate_cut_size(partition_cpu, hyperedge_index_cpu, num_vertices)
    
    partition_sizes = torch.bincount(partition)
    total_hyperedges = hyperedge_index[1].max() + 1
    
    print(f"Partition sizes: {partition_sizes[0].item()} | {partition_sizes[1].item()}")
    print(f"Cut size: {cut_size}")
    print(f"Total hyperedges: {total_hyperedges}")
    print(f"Cut ratio: {cut_size / total_hyperedges:.4f}")
    
    # Plot results
    plot_embedding_histogram(embeddings)


if __name__ == "__main__":
    main()