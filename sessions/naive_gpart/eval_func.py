import torch

def get_partition_assignment(prob):
    """
    prob: [N, 2]，每個 node 屬於 partition 0/1 的機率
    回傳 hard assignment: [N]，每個 node 的 partition（0 或 1）
    """
    return prob.argmax(dim=1)

def partition_size(assign, num_parts=2):
    """
    assign: [N]，每個 node 的 partition（0, 1, ...）
    回傳每個 partition 的 node 數量
    """
    sizes = [(assign == i).sum().item() for i in range(num_parts)]
    return sizes

def cut_size(assign, edge_index):
    """
    assign: [N]，每個 node 的 partition（0, 1, ...）
    edge_index: [2, E]
    回傳跨 partition 的 edge 數量
    """
    src, dst = edge_index
    cut = ((assign[src] != assign[dst])).sum().item()
    return cut//2 #for undirected graph

def eval_partition(prob, edge_index, num_parts=2, verbose=True):
    """
    綜合評估 partition solution，輸出 partition size 與 cut size
    """
    assign = get_partition_assignment(prob)
    sizes = partition_size(assign, num_parts)
    cut = cut_size(assign, edge_index)
    if verbose:
        print(f"Partition sizes: {sizes}")
        print(f"Cut size: {cut}")
    return sizes, cut 


if __name__ == "__main__":
    prob = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    sizes, cut = eval_partition(prob, edge_index)
    print(f"Partition sizes: {sizes}")
    print(f"Cut size: {cut}")