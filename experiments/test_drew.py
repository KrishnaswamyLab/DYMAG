import torch
import torch.nn.functional as F
import argparse
import sys, os

from src.tudataset_kfold_loader import get_tudataset_with_kfold
from src.model_drew import DRew

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Train DRew on TU dataset with k-fold CV")
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--layers', type=int, default=2)
    
    # DRew specific parameters
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--nu', type=int, default=1, help='Delay parameter (1=no delay, higher=more delay)')
    parser.add_argument('--max_distance', type=int, default=5, help='Maximum k-hop distance')
    parser.add_argument('--eps', type=float, default=0.0, help='GIN epsilon parameter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_norm', action='store_true', default=True)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accs = []

    for fold in range(args.folds):
        print(f"\n--- Fold {fold + 1}/{args.folds} ---")
        train_loader, test_loader, in_channels, num_classes = get_tudataset_with_kfold(
            args.dataset, fold_idx=fold, num_folds=args.folds, batch_size=32
        )
        model = DRew(
            in_channels=in_channels,
            hidden_channels=args.hidden,
            out_channels=num_classes,
            num_layers=args.layers,
            nu=args.nu,
            max_distance=args.max_distance,
            eps=args.eps,
            dropout=args.dropout,
            batch_norm=args.batch_norm
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_loader, optimizer, device)
            acc = test(model, test_loader, device)
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

        accs.append(acc)

    avg_acc = sum(accs) / len(accs)
    print(f"\n✅ Average Accuracy over {args.folds} folds: {avg_acc:.4f}")
    print(f"DRew Config: nu={args.nu}, max_distance={args.max_distance}, layers={args.layers}")

if __name__ == "__main__":
    main() 