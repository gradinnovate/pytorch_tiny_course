import torch
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple, Tuple
import torch_geometric as pyg
import scipy.sparse as sp
import os
import warnings

def get_file_directory(file_path: str) -> str:
    """
    Returns the directory of the given file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The directory containing the file.
    """
    return os.path.dirname(os.path.abspath(file_path))


def parse_header(header_tokens: List[str]) -> Tuple[int, int, int, int]:
    """
    Parses the header line of the graph file.

    Args:
        header_tokens (List[str]): The tokens from the header line.

    Returns:
        Tuple[int, int, int, int]: A tuple containing n, m, fmt, and ncon.
    """
    if len(header_tokens) < 2:
        raise ValueError("Header line must contain at least two integers (n and m).")

    n = int(header_tokens[0])  # Number of vertices
    m = int(header_tokens[1])  # Number of edges

    # Initialize fmt and ncon with default values
    fmt = '000'
    ncon = 1

    if len(header_tokens) >= 3:
        fmt_str = header_tokens[2]
        # Assume fmt is provided as a binary string
        fmt = int(fmt_str, 2) if fmt_str.isdigit() and set(fmt_str).issubset({'0', '1'}) else int(fmt_str)

    if len(header_tokens) >= 4:
        ncon = int(header_tokens[3])

    return n, m, fmt, ncon

def determine_feature_flags(fmt: str) -> Tuple[bool, bool, bool]:
    # Ensure fmt is a 3-digit binary string
    if not (len(fmt) == 3 and set(fmt).issubset({'0', '1'})):
        raise ValueError("fmt must be a 3-digit binary string")

    # Determine feature flags based on fmt
    has_edge_weights = fmt[2] == '1'
    has_vertex_weights = fmt[1] == '1'
    has_vertex_sizes = fmt[0] == '1'
    return has_edge_weights, has_vertex_weights, has_vertex_sizes

def parse_graph_file(file_path: str) -> Tuple[Data, np.ndarray]:
    """
    Parses a graph file and converts it into a PyTorch Geometric Data object.

    Args:
        file_path (str): Path to the graph file.

    Returns:
        Data: PyTorch Geometric Data object representing the graph.
    """
    # Initialize variables
    sizes: List[int] = []
    weights: List[List[int]] = []
    edge_set: set = set()
    edge_list: List[Tuple[int, int]] = []
    edge_weights: List[int] = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Remove comment lines and strip whitespace
    lines = [line.strip() for line in lines if not line.strip().startswith('%')]

    if not lines:
        raise ValueError("Graph file is empty or only contains comments.")

    # Parse header line using the separate function
    header = lines[0].split()
    n, m, fmt, ncon = parse_header(header)

    # Determine feature flags based on fmt
    has_edge_weights, has_vertex_weights, has_vertex_sizes = determine_feature_flags(fmt)

    # Initialize adjacency list
    adjacency_list = [[] for _ in range(n)]

    # Process each vertex line
    for i in range(n):
        if 1 + i >= len(lines):
            raise ValueError(f"Expected {n} vertex lines, but got {i}.")

        line = lines[1 + i]
        tokens = line.split()
        ptr = 0

        # Parse vertex size
        if has_vertex_sizes:
            if ptr >= len(tokens):
                raise ValueError(f"Vertex {i+1} is missing size information.")
            s = int(tokens[ptr])
            sizes.append(s)
            ptr += 1
        else:
            sizes.append(1)  # Default size

        # Parse vertex weights
        if has_vertex_weights:
            if ptr + ncon > len(tokens):
                raise ValueError(f"Vertex {i+1} is missing weight information.")
            w = [int(tokens[ptr + j]) for j in range(ncon)]
            weights.append(w)
            ptr += ncon
        else:
            weights.append([1] * ncon)  # Default weights

        # Parse adjacency list
        while ptr < len(tokens):
            v = int(tokens[ptr]) - 1  # Convert to 0-based index
            if v < 0 or v >= n:
                raise ValueError(f"Vertex index {v+1} out of bounds.")
            ptr += 1

            adjacency_list[i].append(v)
            adjacency_list[v].append(i)

            # To avoid duplicate undirected edges, store them in a sorted tuple
            edge = tuple(sorted((i, v)))
            if edge not in edge_set:
                edge_set.add(edge)
                edge_list.append(edge)

    # Calculate common neighbors as edge weights
    for u, v in edge_list:
        common_neighbors = len(set(adjacency_list[u]) & set(adjacency_list[v]))+1
        edge_weights.append(common_neighbors)

    # Verify the number of edges
    if len(edge_list) != m:
        print(f"Warning: Number of unique edges ({len(edge_list)}) does not match m ({m}).")

    # Create edge_index and edge_attr tensors using numpy
    edge_index_np = np.array(edge_list, dtype=np.int64).T
    #edge_index_np = np.hstack((edge_index_np, edge_index_np[[1, 0], :]))  # Add both directions

    edge_attr_np = np.array(edge_weights, dtype=np.float32)
    #edge_attr_np = np.hstack((edge_attr_np, edge_attr_np))  # Add both directions

    edge_index = torch.from_numpy(edge_index_np).contiguous()
    edge_attr = torch.from_numpy(edge_attr_np).contiguous()

    f1, f4 = generalized_eigen_features(edge_index, n)
    f2 = pyg.utils.degree(edge_index[0], num_nodes=n, dtype=torch.float32)
    f2 = f2.reshape(-1, 1)
    f3 = torch.from_numpy(np.arange(n)).float().reshape(-1, 1)
    

    # normalization
    f1_norm = torch.norm(f1, p='fro')
    f2_norm = torch.norm(f2, p='fro')
    f3_norm = torch.norm(f3, p='fro')
    f4_norm = torch.norm(f4, p='fro')
    if f1_norm > 0 and f2_norm > 0:
        f1 =  f1 / f1_norm * f2_norm
    if f3_norm > 0 and f2_norm > 0:
        f3 = f3 / f3_norm * f2_norm
    if f4_norm > 0 and f2_norm > 0:
        f4 = f4 / f4_norm * f2_norm

    x = torch.cat([f1,f2,f3], dim=1)

    # Create and return the Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def generalized_eigen_features(edge_index, num_nodes=None, max_iter=20):
    # Convert edge_index to numpy adjacency matrix
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    temp_edge_list = edge_index.numpy()
    adj = sp.csr_matrix((np.ones(temp_edge_list.shape[1]), 
                        (temp_edge_list[0,:], temp_edge_list[1,:])), 
                        shape=(num_nodes, num_nodes))
    
    # Compute degree matrix
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    
    # Compute normalized Laplacian matrix L = D^(-1/2) A D^(-1/2)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-10))
    L = sp.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    try:
        # Initialize random guess for eigenvectors
        X = np.random.rand(num_nodes, 2)
        X = np.linalg.qr(X)[0]  # Orthonormalize initial vectors
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigenvalues, eigenvectors = sp.linalg.lobpcg(
            L, X, 
            M=D,  
            largest=False,
            maxiter=max_iter,
            tol=1e-4
        )
        
        # Get Fiedler vector (second smallest eigenvector)
        fiedler_vector = eigenvectors[:, 1]
        
        # Partition assignment: make 0/1 as balanced as possible
        idx = np.argsort(fiedler_vector)
        partition = np.ones_like(fiedler_vector, dtype=int)
        partition[idx[:len(idx)//2]] = 0
    
        # Convert to torch tensor and reshape
        #x = torch.from_numpy(partition).float().reshape(-1, 1)
        x = torch.from_numpy(fiedler_vector).float().reshape(-1, 1)
        partition = torch.from_numpy(partition).float().reshape(-1, 1)
        return x, partition
        
    except Exception as e:
        print(f"Warning: Eigenvalue computation failed: {str(e)}")
        return torch.zeros((num_nodes, 1))


# Example usage:
if __name__ == "__main__":
    import time
    base_dir = get_file_directory(__file__)

    start_time = time.time()
    graph_data = parse_graph_file(os.path.join(base_dir, "tiny.graph"))
    end_time = time.time()
    
    device = torch.device('cpu')
    graph_data = graph_data.to(device)
    torch.save(graph_data, os.path.join(base_dir, 'tiny.pt'))

    print("Graph is converted to pytorch geometric data successfully.")
    print("Data details:")
    print(graph_data)
    
    
    
    
