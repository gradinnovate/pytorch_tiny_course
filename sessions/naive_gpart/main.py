import os
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import GraphPartitionNet
from loss_func import total_loss
from eval_func import eval_partition
from util_log import MLFlowLogger


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    torch.manual_seed(3)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, 'config.yaml'))

    # device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # load data
    data = torch.load(os.path.join(base_dir, 'data/tiny.pt'), weights_only=False)
    data = data.to(device)

    # model
    model_cfg = config['model']
    model = GraphPartitionNet(
        input_dim=model_cfg['input_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        output_dim=model_cfg['output_dim'],
        aggr=model_cfg['aggr']
    ).to(device)
    print(f"model: {model}")

    # optimizer
    lr = config['train']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # mlflow logger
    mlflow_cfg = config['mlflow']
    logger = MLFlowLogger(
        tracking_uri=mlflow_cfg['tracking_uri'],
        experiment_name=mlflow_cfg['experiment_name'],
        params={**model_cfg, **config['train']}
    )

    # training
    epochs = config['train']['epochs']
    alpha = config['train'].get('alpha', 1.0)
    beta = config['train'].get('beta', 1.0)
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        prob = model(data.x, data.edge_index)
        loss_dict = total_loss(prob, data.edge_index, None, alpha=alpha, beta=beta)
        loss = loss_dict['loss']
        ncut = loss_dict['ncut']
        bal = loss_dict['balance']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()
        logger.log_loss(loss.item(), step=epoch)
        logger.log_metrics({'ncut': ncut, 'balance': bal}, step=epoch)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, ncut: {ncut:.4f}, balance: {bal:.4f}")

    logger.end()

    # evaluation
    model.eval()
    with torch.no_grad():
        prob = model(data.x, data.edge_index)
        print("\nFinal partition evaluation:")
        eval_partition(prob, data.edge_index, num_parts=model_cfg['output_dim'])

if __name__ == "__main__":
    main() 