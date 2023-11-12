import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LatticePhaseNet
from data import LatticeGen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_example = 10

q_abc_res   = 100
q_ang_res   = 90

embed_layer = 10
embed_dim   = 100
latent_dim  = 100
hidden_layer= 10
hidden_dim  = 100
output_dim  = q_abc_res * q_ang_res

loss_fn = nn.CrossEntropyLoss()
lr = 0.001
weight_decay = 0.05

model = LatticePhaseNet(embed_layer, embed_dim, latent_dim, hidden_layer, hidden_dim, output_dim).to(device).to(torch.float64)

optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

lattice_gen = LatticeGen(q_abc_res, q_ang_res)

for n in range(num_example):
    x, y = lattice_gen.gen_valid_random_data()
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    pred_y = model(x)
    loss = loss_fn(pred_y, y.flatten())
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()