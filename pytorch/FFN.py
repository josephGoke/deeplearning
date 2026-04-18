import torch
import torch.nn as nn




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Model
class FCN(nn.Module):
    '''
        Feed Forward Fully-connected Neural Network

        layer_sizes: list of layer sizes including input and output layers
        activation: activation function ('relu', 'tanh', 'sigmoid', 'leakyrelu')
    '''

    def __init__(self, layer_sizes: list, activation='relu'):

        super(FCN, self).__init__()
        activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'leakyrelu': nn.LeakyReLU()}
        self.activation = activation_map[activation]
        
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(self.activation)


        #Sequential net
        self.net = nn.Sequential(*layers)

    def forward(self, x, device):
        super().to(device)
        return self.net(x)
    


class FCN_Dropout(nn.Module):
    '''
        Feed Forward Fully-connected Neural Network with Dropout & BatchNormalization
    '''
    def __init__(self, layer_sizes, dropout_rate=0.5, activation='relu') -> None:
        super(FCN_Dropout, self).__init__()

        activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'leakyrelu': nn.LeakyReLU()}
        self.activation = activation_map[activation]
        self.dropout = nn.Dropout(dropout_rate)

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                
                layers.append(self.activation)
                layers.append(self.dropout)

        self.net = nn.Sequential(*layers)

    def forward(self, x, device=device):
        super().to(device)
        return self.net(x)
    
