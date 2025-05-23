import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 n_channels = 1,
                 bce = False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        # Build a feature extractor
        self.feature = nn.Sequential(
            nn.Conv2d(n_channels, 8, 3, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # Build a classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.Linear(32, num_classes),
            # nn.Softmax(dim=1)
            )
        if bce:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.feature(x)
        pred = self.classifier(features)
        return pred

    def loss_function(self, pred, **kwargs) -> dict:
        y_true = kwargs['labels']
        losses = {}
        loss = self.criterion(pred.to(torch.float), y_true.to(torch.float))
        losses['loss'] = loss
        return losses
