import torch
import torch.nn.functional as F
import argparse
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tudataset_kfold_loader import get_tudataset_with_kfold
from src.model_grandpp import GRANDPP

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        logits, reg_loss = model(data, train_mode=True)
        loss = F.cross_entropy(logits, data.y) + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data, train_mode=False)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Train GRAND++ on TU dataset")
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dropnode', type=float, default=0.5)
    parser.add_argument('--views', type=int, default=4)
    parser.add_argument('--lambda_cons', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accs = []

    for fold in range(args.folds):
        print(f"\n--- Fold {fold + 1}/{args.folds} ---")
        train_loader, test_loader, in_channels, num_classes = get_tudataset_with_kfold(
            args.dataset, fold_idx=fold, num_folds=args.folds, batch_size=32
        )
        model = GRANDPP(
            in_channels=in_channels,
            hidden_channels=64,
            out_channels=num_classes,
            K=args.K,
            dropout=args.dropout,
            dropnode_rate=args.dropnode,
            n_views=args.views,
            lam_consistency=args.lambda_cons
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_loader, optimizer, device)
            acc = test(model, test_loader, device)
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

        accs.append(acc)

    avg_acc = sum(accs) / len(accs)
    print(f"\n✅ Average Accuracy over {args.folds} folds: {avg_acc:.4f}")

if __name__ == "__main__":
    main()