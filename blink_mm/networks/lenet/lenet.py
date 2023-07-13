import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, vis_features=False):
        super().__init__()
        self.vis_features = vis_features

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        if self.vis_features:
            self.fc1 = nn.Linear(120, 84)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(84, 2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(2, num_classes)
        else:
            self.fc1 = nn.Linear(120, 84)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        if self.vis_features:
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
        else:
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
        return out


def lenet(**kwargs):
    return LeNet5(**kwargs)
