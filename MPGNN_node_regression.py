import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from wave_scattering import *

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
    
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
# load a graph
dataset = dgl.data.CiteseerGraphDataset()
graph = dataset[0]

# wave scattering params
c = 0.5                # wave velocity
num_tp = 20            # num. time points
num_init_cond = 10     # num. initial conditions

# analytically solve the wave eqn
solve_wave_eqn(graph, c, num_tp, num_init_cond)

# MPGNN params
num_epochs = 100

node_features = graph.ndata['wave_eqn_soln']
node_labels = graph.ndata['node_props']
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']

n_features = node_features.shape[1]
n_labels = node_labels[0].shape[1]

model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
opt = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    
    model.train()
    
    # forward propagation by using all nodes
    pred = model(graph, node_features)
    
    # compute loss
    loss = F.mse_loss(pred[train_mask], node_labels[train_mask])
    
    # compute validation accuracy
    acc = evaluate(model, graph, node_features, node_labels, valid_mask)
    
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if epoch % 10 == 0:
        print('In epoch {}, loss: {:.3f}, val acc: {:.3f}'.format(epoch, loss, acc))