import torch

def parse_hgr_file(file_path):
    """
    Parse .hgr file and convert to hyperedge_index tensor format for PyTorch Geometric.
    
    Args:
        file_path (str): Path to .hgr file
        
    Returns:
        torch.Tensor: hyperedge_index tensor of shape (2, num_connections)
                     where first row contains node indices and second row contains hyperedge indices
    """
    node_indices = []
    hyperedge_indices = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Parse header (line 1: num_hyperedges num_vertices)
        header = lines[0].strip().split()
        num_hyperedges = int(header[0])
        num_vertices = int(header[1])
        
        # Parse hyperedges (lines 2 onwards)
        for hyperedge_id, line in enumerate(lines[1:]):
            vertices = list(map(int, line.strip().split()))
            
            # Add each vertex in this hyperedge
            for vertex_id in vertices:
                node_indices.append(vertex_id - 1)  # Convert to 0-based indexing
                hyperedge_indices.append(hyperedge_id)
    
    # Create hyperedge_index tensor
    hyperedge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)
    
    return hyperedge_index, num_vertices, num_hyperedges


if __name__ == "__main__":
    # Test with ibm01.hgr
    file_path = "ibm01.hgr"
    
    print(f"Parsing {file_path}...")
    hyperedge_index, num_vertices, num_hyperedges = parse_hgr_file(file_path)
    
    print(f"Number of vertices: {num_vertices}")
    print(f"Number of hyperedges: {num_hyperedges}")
    print(f"Hyperedge index shape: {hyperedge_index.shape}")
    print(f"First 10 connections:")
    print(f"Node indices: {hyperedge_index[0][:10]}")
    print(f"Hyperedge indices: {hyperedge_index[1][:10]}")
    
    # Verify first hyperedge
    first_hyperedge_nodes = hyperedge_index[0][hyperedge_index[1] == 0]
    print(f"First hyperedge contains nodes: {first_hyperedge_nodes} (0-indexed)")
    print(f"First hyperedge contains nodes: {first_hyperedge_nodes + 1} (1-indexed)")