import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class WrapedVGG(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 freeze_layers: bool = False,
                 bce = False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.transform = VGG16_BN_Weights.IMAGENET1K_V1.transforms()
        self.net = vgg16_bn(VGG16_BN_Weights.IMAGENET1K_V1)
        if freeze_layers:
            for params in self.net.parameters():
                params.requires_grad = False
        in_features = self.net._modules['classifier'][-1].in_features
        self.net._modules['classifier'][-1] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        for params in self.net.classifier.parameters():
            params.requires_grad = True
        if bce:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.transform(x)
        pred = self.net(x)
        return pred
    
    def loss_function(self, pred, **kwargs) -> dict:
        y_true = kwargs['labels']
        losses = {}
        loss = self.criterion(pred.to(torch.float), y_true.to(torch.float))
        losses['loss'] = loss
        return losses
    
    def load_checkpoint(self, root, map_location = None):
        new_weights = self.net.state_dict()
        old_weights = list(torch.load(root, map_location = map_location)['state_dict'].items())
        i=0
        for k, _ in new_weights.items():
            new_weights[k] = old_weights[i][1]
            i += 1
        self.net.load_state_dict(new_weights)
    
    def freeze_encoder(self):
        for name, params in self.net.named_parameters():
            if 'classifier' not in name:
                params.requires_grad = False

if __name__ == '__main__':
    wraped_net = WrapedVGG()
    print(wraped_net)
    data = torch.randn(1, 3, 128, 128) 
    result = wraped_net(data)
    print(result)