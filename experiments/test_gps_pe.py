import torch
import torch.nn.functional as F
import argparse

from src.tudataset_kfold_loader import get_tudataset_with_kfold_pe
from src.model_gps_pe import GraphGPS

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        lap_pe = getattr(data, 'lap_pe', None)
        rw_pe  = getattr(data, 'rw_pe',  None)
        out  = model(data.x, data.edge_index, data.batch, lap_pe=lap_pe, rw_pe=rw_pe)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        lap_pe = getattr(data, 'lap_pe', None)
        rw_pe  = getattr(data, 'rw_pe',  None)
        out  = model(data.x, data.edge_index, data.batch, lap_pe=lap_pe, rw_pe=rw_pe)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# --------------------------------------------------------------------- #
# Main k-fold loop
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="GraphGPS on TU dataset with k-fold CV")
    p.add_argument("--dataset", type=str, default="PROTEINS")
    p.add_argument("--folds",   type=int, default=10)
    p.add_argument("--epochs",  type=int, default=20)
    p.add_argument("--layers",  type=int, default=2)
    p.add_argument("--hidden",  type=int, default=64)
    p.add_argument("--heads",   type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accs   = []

    for fold in range(args.folds):
        print(f"\n--- Fold {fold+1}/{args.folds} ---")

        train_loader, test_loader, in_dim, n_classes = get_tudataset_with_kfold_pe(
            args.dataset, fold_idx=fold, num_folds=args.folds, batch_size=32)

        model = GraphGPS(in_dim, args.hidden, n_classes,
                         num_layers=args.layers, heads=args.heads).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_loader, optim, device)
            acc  = test(model,  test_loader,  device)
            print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Test Acc {acc:.4f}")

        accs.append(acc)

    print(f"\n✅ Avg accuracy over {args.folds} folds: "
          f"{sum(accs)/len(accs):.4f}")

if __name__ == "__main__":
    main()