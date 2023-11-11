import torch.nn as nn
import torch.nn.functional as F

class LatticePhaseLoss():
    pass

class PeakEmbedNet(nn.Module):
    def __init__(self, embed_layer, embed_dim, latent_dim):
        super(PeakEmbedNet, self).__init__()
        self.input = nn.Linear(1, embed_dim)
        self.embed = [nn.Linear(embed_dim, embed_dim) for _ in range(embed_layer)]
        self.latent = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        out = F.relu(self.input(x))
        for layer in self.embed:
            out = F.relu(layer(out))
        return F.relu(self.latent(out))


class LatticePhaseNet(nn.Module):
    def __init__(self, embed_layer, embed_dim, latent_dim, hidden_layer, hidden_dim, output_dim):
        super(LatticePhaseNet, self).__init__()
        self.peakembednet = PeakEmbedNet(embed_layer, embed_dim, latent_dim)
        self.lin = nn.Linear(embed_dim, hidden_dim)
        self.hidden = [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layer)]
        self.output = nn.linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x_embeded = sum([self.peakembednet(peak) for peak in x])
        out = F.relu(self.lin(x_embeded))
        for layer in self.hidden:
            out = F.relu(layer(out))
        return F.softmax(self.output(out))