import torch
from torch import nn

# Define the Classifier3D model structure (same as in the original file)
class Classifier3D(nn.Module):
    def __init__(self, stages, pooling_configs, num_classes=11, desired_res=(32, 32, 32)):
        super(Classifier3D, self).__init__()

        self.features = nn.Sequential()

        for i, stage in enumerate(stages):
            self.features.add_module(f"stage_{i}", self.conv_stage(stage))

            if i < len(pooling_configs):
                pool_layer = self.create_pooling_layer(pooling_configs[i])
                if pool_layer:
                    self.features.add_module(f"pool_{i}", pool_layer)

        self._to_linear = None
        self._get_conv_output((1, *desired_res))

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def conv_stage(self, layer_configs):
        layers = []
        for config in layer_configs:
            in_channels, out_channels, kernel_size, stride, padding = config
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def create_pooling_layer(self, config):
        if config is None:
            return None
        kernel_size, stride, padding = config
        return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.features(input)
        self._to_linear = int(torch.prod(torch.tensor(output_feat.size()[1:])))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ClassifierFC_spec(nn.Module):
    def __init__(self, input_size, hidden_sizes1, hidden_sizes2, hidden_sizes3, num_hidden, num_classes, dropout_rate=0):
        super(ClassifierFC_spec, self).__init__()
        layers = []

        # Input layer with batch normalization and dropout
        layers.append(nn.Linear(input_size, hidden_sizes1, bias=True))
        layers.append(nn.BatchNorm1d(hidden_sizes1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))  # Dropout

        # Second layer with batch normalization and dropout
        layers.append(nn.Linear(hidden_sizes1, hidden_sizes2, bias=True))
        layers.append(nn.BatchNorm1d(hidden_sizes2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))  # Dropout

        # Hidden layers with batch normalization and dropout
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_sizes2, hidden_sizes2, bias=True))
            layers.append(nn.BatchNorm1d(hidden_sizes2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))  # Dropout

        # Output layers with batch normalization and dropout
        layers.append(nn.Linear(hidden_sizes2, hidden_sizes3, bias=True))
        layers.append(nn.BatchNorm1d(hidden_sizes3))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))  # Dropout

        # Final output layer (no dropout here)
        layers.append(nn.Linear(hidden_sizes3, num_classes))

        # Register all layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def conv_stage_2d(layer_configs):
    layers = []
    for config in layer_configs:
        in_channels, out_channels, kernel_size, stride, padding = config
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# Helper function to create a pooling layer
def create_pooling_layer_2d(config):
    if config is None:
        return None
    kernel_size, stride, padding = config
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

# 2D Convolutional Classifier
class Classifier2D(nn.Module):
    def __init__(self, stages, pooling_configs, num_classes=11, shape_X_l=7, shape_X_p=13):
        super(Classifier2D, self).__init__()

        self.shape_X_l = shape_X_l
        self.shape_X_p = shape_X_p

        self.features = nn.Sequential()

        # Add convolutional stages and pooling layers
        for i, stage in enumerate(stages):
            self.features.add_module(f"stage_{i}", conv_stage_2d(stage))

            # Add MaxPooling layer after each stage based on the pooling configuration
            if i < len(pooling_configs):
                pool_layer = create_pooling_layer_2d(pooling_configs[i])
                if pool_layer:
                    self.features.add_module(f"pool_{i}", pool_layer)

        # Calculate the size of the flattened features after the conv layers
        self._to_linear = None
        self._get_conv_output((1, shape_X_l, shape_X_p))  # Add a channel dimension here

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)

    # Helper function to calculate the output size after convolution and pooling
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.features(input)
        self._to_linear = int(torch.prod(torch.tensor(output_feat.size()[1:])))

    # Optional: Initialize weights with specific methods
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # Forward pass
    def forward(self, x):
        # Reshape input from (batch, shape_X_l * shape_X_p) -> (batch, 1, shape_X_l, shape_X_p)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.shape_X_l, self.shape_X_p)  # Add a channel dimension

        # Pass through convolutional layers
        x = self.features(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x
