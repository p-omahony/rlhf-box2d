import torch.nn as nn

def nn_constructor(cfg):
    layers = []
    for block in cfg:
        for layer in cfg[block]:
            layer_name = layer[0]
            if layer_name == 'Linear':
                layers.append(nn.Linear(layer[1], layer[2]))
            elif layer_name == 'Dropout':
                layers.append(nn.Dropout(layer[1]))
            elif layer_name == 'PReLU':
                layers.append(nn.PReLU())
            elif layer_name == 'ReLU':
                layers.append(nn.ReLU())
            else:
                raise ValueError("Layer name not written correctly or not implemented in utils.functions.nn_config_constructor")
    return layers

class MultiLayerPerceptron(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        layers = nn_constructor(cfg)
        
        self.fc = nn.Sequential(
            *layers
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x