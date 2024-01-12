    ## FIXME: reset_params in cross_validate: is it really resetting? possible data leak!!!

"""
train function for the classifier using the PDE's for graph classification.

Xingzhi Sun
April 2023

"""
import torch
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader as DataListLoader
from sklearn.model_selection import KFold
import time
import pandas as pd

def train(model, dataloader, optimizer, criterion, device, num_epochs):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(dataloader):
            # Get data and labels, and move them to the device
            x, edge_index, edge_weight, batch, y = data.x.to(device), data.edge_index.to(device), \
                                                   data.edge_weight.to(device), data.batch.to(device), \
                                                   data.y.to(device)
            
            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            outputs = model(x, edge_index, edge_weight, batch)
            
            # Compute loss
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        epoch_loss = running_loss / (i + 1)
        epoch_acc = correct / total
        if epoch % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

def test(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for i, data in enumerate(dataloader):
            # Get data and labels, and move them to the device
            x, edge_index, edge_weight, batch, y = data.x.to(device), data.edge_index.to(device), \
                                                   data.edge_weight.to(device), data.batch.to(device), \
                                                   data.y.to(device)

            output = model(x, edge_index, edge_weight, batch)  # Forward pass through the model
            _, predicted = torch.max(output, 1)  # Get the predicted class for each example
            total += y.size(0)  # Increment the total count
            correct += (predicted == y).sum().item()  # Increment the correct count
            
    accuracy = correct / total  # Calculate accuracy
    print(f'Test accuracy: {accuracy * 100:.2f}%')


def train1epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    for data in dataloader:
        # Get data and labels, and move them to the device
        x, edge_index, edge_weight, batch, y = data.x.to(device), data.edge_index.to(device), \
                                               data.edge_weight.to(device), data.batch.to(device), \
                                               data.y.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(x, edge_index, edge_weight, batch)

        # Compute loss
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    return running_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            x, edge_index, edge_weight, batch, y = data.x.to(device), data.edge_index.to(device), \
                                                   data.edge_weight.to(device), data.batch.to(device), \
                                                   data.y.to(device)

            outputs = model(x, edge_index, edge_weight, batch)

            loss = criterion(outputs, y)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return running_loss / len(dataloader), correct / total

def cross_validate(model, dataset, n_splits, batch_size, num_epochs, lr, weight_decay, device):
    """
    Generate the folds and do CV on one model.

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        n_splits (_type_): _description_
        batch_size (_type_): _description_
        num_epochs (_type_): _description_
        lr (_type_): _description_
        weight_decay (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    id_folds = kf.split(dataset)
    return cross_validate_given_id_folds(model, dataset, id_folds, batch_size, num_epochs, lr, weight_decay, device)

def cross_validate_given_id_folds(model, dataset, id_folds, batch_size, num_epochs, lr, weight_decay, device):
    """Do CV on one model given the list of (train_idx, val_idx) in each folds.
    Using this function, you can fix the indices to be the same for input and run on different the models.

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        id_folds (_type_): _description_
        batch_size (_type_): _description_
        num_epochs (_type_): _description_
        lr (_type_): _description_
        weight_decay (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    epoch_data = dict()
    for fold, (train_idx, val_idx) in enumerate(id_folds):
        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]

        train_loader = DataListLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataListLoader(val_set, batch_size=batch_size, shuffle=False)

        # Reset the model weights and optimizer state
        ## FIXME: reset_params in cross_validate: is it really resetting? possible data leak!!!
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        start_time = time.time()
        # Create a DataFrame for the epoch data for this fold
        fold_epoch_data = pd.DataFrame(columns=['epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy'])
        for epoch in range(num_epochs):
            train_loss, train_acc = train1epoch(model, train_loader, optimizer, criterion, device)
            # Record the epoch data
            fold_epoch_data = fold_epoch_data.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
            }, ignore_index=True)

        training_time = time.time() - start_time
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_data[fold] = fold_epoch_data
        results.append({
            'fold': fold,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'validation_loss': val_loss,
            'validation_accuracy': val_acc,
            'training_time': training_time,
        })

    results_df = pd.DataFrame(results)
    return results_df, epoch_data