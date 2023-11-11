import torch

from model import LatticePhaseNet, LatticePhaseLoss
from data import gen_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_layer = 1
embed_dim   = 10
latent_dim  = 10
hidden_layer= 2
hidden_dim  = 10
output_dim  = 9000

learning_rate = 0.001

lossfn = LatticePhaseLoss()

model = LatticePhaseNet(embed_layer, embed_dim, latent_dim, hidden_layer, hidden_dim, output_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):

    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    print(f'[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)