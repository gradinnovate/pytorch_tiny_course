import torch
import torch_geometric as pyg

def norm_cut_loss(y_pred, edge_index):
    y = y_pred
    d = pyg.utils.degree(edge_index[0], num_nodes=y.size(0))
    gamma = y.t() @ d
   
    c = y[edge_index[0],0]* y[edge_index[1], 1]
    c = torch.sum(c)
  
    
    return torch.sum(torch.div(c, gamma)) 


def area_balance_loss(y_pred, areas):
    y = y_pred 
    num_nodes = y.shape[0]
    avg = torch.sum(areas).cpu()/y.shape[1]
    gamma = y.t() @ areas
    diff = gamma - avg
    return torch.sum(torch.pow(diff,2))/(num_nodes**2)


def total_loss(y_pred, edge_index, areas=None, alpha=1.0, beta=1.0):
    """
    y_pred: [N, 2]
    edge_index: [2, E]
    areas: [N] or None (若 None 則全 1)
    alpha: norm_cut_loss 權重
    beta: area_balance_loss 權重
    """
    if areas is None:
        areas = torch.ones(y_pred.size(0), device=y_pred.device)
    ncut = norm_cut_loss(y_pred, edge_index)
    bal = area_balance_loss(y_pred, areas)
    loss = alpha * ncut + beta * bal
    return {
        'loss': loss,
        'ncut': ncut.item(),
        'balance': bal.item()
    }


if __name__ == "__main__":
    y_pred = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    areas = torch.ones(2)
    loss = area_balance_loss(y_pred, areas)
    print(loss)
    
    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    loss = norm_cut_loss(y_pred, edge_index)
    print(loss)
