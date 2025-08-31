import torch

def parse_hgr_file(file_path):
    """
    Parse .hgr file and convert to hyperedge_index tensor format for PyTorch Geometric.
    Also compute hyperedge weights as 1/|e| where |e| is the number of nodes in each hyperedge.
    
    Args:
        file_path (str): Path to .hgr file
        
    Returns:
        tuple: (hyperedge_index, hyperedge_weights, num_vertices, num_hyperedges)
            - hyperedge_index: tensor of shape (2, num_connections)
            - hyperedge_weights: tensor of shape (num_hyperedges,) containing 1/|e| for each hyperedge
            - num_vertices: int, number of vertices
            - num_hyperedges: int, number of hyperedges
    """
    node_indices = []
    hyperedge_indices = []
    hyperedge_weights = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Parse header (line 1: num_hyperedges num_vertices)
        header = lines[0].strip().split()
        num_hyperedges = int(header[0])
        num_vertices = int(header[1])
        
        # Parse hyperedges (lines 2 onwards)
        for hyperedge_id, line in enumerate(lines[1:]):
            vertices = list(map(int, line.strip().split()))
            hyperedge_size = len(vertices)
            hyperedge_weights.append(hyperedge_size)  # Weight = |e|
            
            # Add each vertex in this hyperedge
            for vertex_id in vertices:
                node_indices.append(vertex_id - 1)  # Convert to 0-based indexing
                hyperedge_indices.append(hyperedge_id)
    
    # Create tensors
    hyperedge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)
    hyperedge_weights = torch.tensor(hyperedge_weights, dtype=torch.float)
    
    return hyperedge_index, hyperedge_weights, num_vertices, num_hyperedges


if __name__ == "__main__":
    # Test with ibm01.hgr
    file_path = "ibm01.hgr"
    
    print(f"Parsing {file_path}...")
    hyperedge_index, hyperedge_weights, num_vertices, num_hyperedges = parse_hgr_file(file_path)
    
    print(f"Number of vertices: {num_vertices}")
    print(f"Number of hyperedges: {num_hyperedges}")
    print(f"Hyperedge index shape: {hyperedge_index.shape}")
    print(f"Hyperedge weights shape: {hyperedge_weights.shape}")
    print(f"Hyperedge weight stats - Mean: {hyperedge_weights.mean():.4f}, Min: {hyperedge_weights.min():.4f}, Max: {hyperedge_weights.max():.4f}")
    print(f"First 10 connections:")
    print(f"Node indices: {hyperedge_index[0][:10]}")
    print(f"Hyperedge indices: {hyperedge_index[1][:10]}")
    print(f"First 10 hyperedge weights: {hyperedge_weights[:10]}")
    
    # Verify first hyperedge
    first_hyperedge_nodes = hyperedge_index[0][hyperedge_index[1] == 0]
    first_hyperedge_size = len(first_hyperedge_nodes)
    expected_weight = 1.0 / first_hyperedge_size
    print(f"First hyperedge contains nodes: {first_hyperedge_nodes} (0-indexed)")
    print(f"First hyperedge contains nodes: {first_hyperedge_nodes + 1} (1-indexed)")
    print(f"First hyperedge size: {first_hyperedge_size}, weight: {hyperedge_weights[0]:.4f} (should be {expected_weight:.4f})")