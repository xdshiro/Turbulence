import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
from scipy.optimize import brute
# import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import collections
import itertools

# from sklearn.model_selection import GridSearchCV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

DataInList = ['AnapoleFPInput_8nmPixel_full.npy', 'AnapoleFPInput_8nmPixel.npy']
DataInCoeff = ['data_structure_coeff.npy', 'data_structure_coeff_short_500.npy']
DataInHeights = ['data_heights.npy', 'data_heights_short_500.npy']
DataOutList = ['AnapoleFPOutput_8nmPixel_full.npy', 'AnapoleFPOutput_8nmPixel.npy']
# savename=DataInList[0][0:-4]
data_images_heights = np.load('../../Dataset/Anapole/' + DataInList[0], allow_pickle=True)
data_heights = np.load('../../Dataset/Anapole/' + DataInHeights[0])
data_coeff = np.load('../../Dataset/Anapole/' + DataInCoeff[0])
data_moments = np.load('../../Dataset/Anapole/' + DataOutList[0], allow_pickle=True)
# data_images_heights = np.load('../Dataset/Anapole/' + DataInList[0], allow_pickle=True)
# data_heights = np.load('../Dataset/Anapole/' + DataInHeights[0])
# data_coeff = np.load('../Dataset/Anapole/' + DataInCoeff[0])
# data_moments = np.load('../Dataset/Anapole/' + DataOutList[0], allow_pickle=True)
row_norms = np.linalg.norm(data_moments, axis=1)
X = np.hstack((data_coeff, data_heights.reshape(-1, 1)))
y = data_moments
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)
# X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size=0.2, random_state=42)
X_train, X_val_test, y_train, y_val_test = train_test_split(X_torch, y_torch, test_size=0.3, random_state=37)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=37)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


# X_train, X_val_test, y_train, y_val_test = train_test_split(X_torch, y_torch, test_size=0.3, random_state=37)
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=37)

class AdvancedNet(nn.Module):
    def __init__(self, input_size, output_size, layer_sizes, dropout_rate):
        super(AdvancedNet, self).__init__()
        layers = []

        # Input layer
        prev_size = input_size

        # Add hidden layers dynamically based on layer_sizes
        for idx, size in enumerate(layer_sizes):
            if idx % 2 == 0:  # Adding residual connections every two layers, but not at the start
                layers.append(
                    ResidualBlock(prev_size, size, dropout_rate))  # Update ResidualBlock with input and output sizes
                prev_size = size  # Update prev_size after the residual block
            else:
                layers.append(nn.Linear(prev_size, size, bias=False))
                layers.append(nn.BatchNorm1d(size))
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = size  # Update prev_size for the next layer

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        # ModuleList of all layers
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        identity = x
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, identity)  # Use identity only in residual blocks
                identity = x  # Update identity after each residual block
            else:
                x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.norm = nn.BatchNorm1d(output_size)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Transform for identity if input and output sizes are different
        if input_size != output_size:
            self.identity_transform = nn.Linear(input_size, output_size)
        else:
            self.identity_transform = None

    def forward(self, x, identity):
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.identity_transform is not None:
            identity = self.identity_transform(identity)
        out += identity

        return out


def loop_train(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    total_loss = 0
    for i, (inputs, targets) in enumerate(train_loader, 1):  # Start enumeration from 1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        total_loss += loss.item()  # Accumulate the loss

    return total_loss / len(train_loader)  # Return the average loss


def loop_test(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to track the gradients
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute the loss
            total_loss += loss.item()  # Accumulate the loss
    return total_loss / len(test_loader)  # Return the average loss


def plot_losses(train_losses, test_losses, decimals=3):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses,
             label=f'Train Loss {train_losses[-1]: .{decimals}f} (min: {min(train_losses): .{decimals}f})')
    plt.plot(test_losses, label=f'Test Loss {test_losses[-1]: .{decimals}f} (min: {min(test_losses): .{decimals}f})')
    plt.title('Training and Testing Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


hyperparams = {
    'input_size': X_train.shape[1],
    'output_size': y_train.shape[1],
    # 'layer_sizes': [128, 256, 512, 1024, 2048, 2048, 1024, 512, 256, 128],  # Control number and size of hidden layers
    'layer_sizes': [256, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 512, 256],
    # Control number and size of hidden layers
    # 'layer_sizes': [512, 512],  # Control number and size of hidden layers
    # 'layer_sizes': [128, 256, 512, 256, 128],  # Control number and size of hidden layers
    # 'layer_sizes': [128, 256, 128],  # Control number and size of hidden layers
    'dropout_rate': 0.3,  # Control dropout rate
    'learning_rate': 1e-3,  # Control learning rate
    'patience': 15,  # Number of epochs between learning rate decay
    'factor': 0.2  # Multiplicative factor of learning rate decay
}

# model = torch.nn.DataParallel(
#     AdvancedNet(
#         input_size=hyperparams['input_size'],
#         output_size=hyperparams['output_size'],
#         layer_sizes=hyperparams['layer_sizes'],
#         dropout_rate=hyperparams['dropout_rate']
#     )
# ).to(device)
#
# criterion = nn.MSELoss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=hyperparams['factor'], patience=hyperparams['patience'],
#                               verbose=True)

num_epochs = 100

param_grid = {
    'batch_size': [32, 64, 128],
    'learning_rate': [1e-2, 1e-3, 1e-4],
    'dropout_rate': [0.5]
}

param_grid = {
    'batch_size': [128],
    'learning_rate': [1e-3],
    'dropout_rate': [0]
}

keys, values = zip(*param_grid.items())
print_every = 10
for combination in itertools.product(*values):
    name_extra = 'no_bias_skip_2nd_initial'
    params = dict(zip(keys, combination))
    print(params)
    # batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
    hyperparams = {
        'input_size': X_train.shape[1],
        'output_size': y_train.shape[1],
        # 'layer_sizes': [128, 256, 512, 1024, 2048, 2048, 1024, 512, 256, 128],  # Control number and size of hidden layers
        'layer_sizes': [256, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 512, 256],
        # Control number and size of hidden layers
        # 'layer_sizes': [512, 512],  # Control number and size of hidden layers
        # 'layer_sizes': [128, 256, 512, 256, 128],  # Control number and size of hidden layers
        # 'layer_sizes': [128, 256, 128],  # Control number and size of hidden layers
        'dropout_rate': params['dropout_rate'],  # Control dropout rate
        'learning_rate': params['learning_rate'],  # Control learning rate
        'patience': 15,  # Number of epochs between learning rate decay
        'factor': 0.2  # Multiplicative factor of learning rate decay
    }

    model = torch.nn.DataParallel(
        AdvancedNet(
            input_size=hyperparams['input_size'],
            output_size=hyperparams['output_size'],
            layer_sizes=hyperparams['layer_sizes'],
            dropout_rate=hyperparams['dropout_rate']
        )
    ).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=hyperparams['factor'], patience=hyperparams['patience'],
                                  verbose=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_losses.append(loop_train(model, train_loader, criterion, optimizer))
        val_losses.append(loop_test(model, val_loader, criterion))

        scheduler.step(val_losses[-1])

        if (epoch + 1) % print_every == 0:
            print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f} ({name_extra})')

        # Save model and losses every 50 epochs
        if (epoch + 1) % 100 == 0:
            # Save the model state
            print(f'model_epoch_{epoch + 1}.pth was saved')
            name = (
                f'batch={params["batch_size"]}_lr={hyperparams["learning_rate"]}_drop={hyperparams["dropout_rate"]}'
                f'_{name_extra}_'
            )
            torch.save(model.state_dict(), f'model_epoch_{epoch + 1}_{name}.pth')
            # Save losses
            with open(f'losses_epoch_{epoch + 1}_{name}.txt', 'w') as f:
                f.write(f'Train Losses: {train_losses}\n')
                f.write(f'Validation Losses: {val_losses}\n')

# Optionally, save the final model and losses
# torch.save(model.state_dict(), 'final_model.pth')
# with open('final_losses.txt', 'w') as f:
#     f.write(f'Train Losses: {train_losses}\n')
#     f.write(f'Validation Losses: {val_losses}\n')
