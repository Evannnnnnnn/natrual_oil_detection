from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(256, output_size),  # Output layer (no BN, no Sigmoid)
            nn.softplus()
        )

    def forward(self, x):
        return self.net(x)
    
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, output_size)  # Output layer (no BN, no Sigmoid)
        )

    def forward(self, x):
        return self.model(x)