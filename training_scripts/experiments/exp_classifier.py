from torch import optim
import pytorch_lightning as pl
from torch.nn import functional as F
import torch
import torchmetrics

class ExpClassifier(pl.LightningModule):
    def __init__(self,
                 cls_model,
                 params: dict) -> None:
        super().__init__()
        self.automatic_optimization = True # manual optim

        self.model = cls_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.train_acc = torchmetrics.classification.Accuracy(task=params["task"],num_classes=self.model.num_classes, num_labels = self.model.num_classes, threshold = 0.5) 
        self.valid_acc = torchmetrics.classification.Accuracy(task=params["task"],num_classes=self.model.num_classes, num_labels = self.model.num_classes, threshold = 0.5) 
        self.train_auc = torchmetrics.classification.AUROC(task=params["task"],num_classes=self.model.num_classes, num_labels = self.model.num_classes) 
        self.valid_auc = torchmetrics.classification.AUROC(task=params["task"],num_classes=self.model.num_classes, num_labels = self.model.num_classes) 

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input, **kwargs):
        return self.model(input, kwargs['labels'])

    def training_step(self, batch, batch_idx):#, optimizer_idx = 0):
        real_img, labels = batch
        if labels.dim() == 1:
            labels = F.one_hot(labels.to(int), self.model.num_classes) if self.model.num_classes > 0 else None
        self.curr_device = real_img.device

        pred = self.model.forward(real_img)
        train_loss = self.model.loss_function(pred, 
                                              labels = labels,
                                              batch_idx = batch_idx)
        if self.params["task"] == 'multiclass':
            self.train_acc(torch.argmax(pred, dim = 1), torch.argmax(labels, dim = 1))
            self.train_auc(pred, torch.argmax(labels, dim = 1))
        else:
            self.train_acc(pred, labels)
            self.train_auc(pred, labels)
        # Record the losses
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_auc', self.train_auc, on_step=True, on_epoch=False)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):#, optimizer_idx = 0):
        real_img, labels = batch
        if labels.dim() == 1:
            labels = F.one_hot(labels.to(int), self.model.num_classes) if self.model.num_classes > 0 else None
        self.curr_device = real_img.device

        pred = self.model.forward(real_img)
        val_loss = self.model.loss_function(pred, 
                                            labels = labels,
                                            batch_idx = batch_idx)
        if self.params["task"] == 'multiclass':
            self.valid_acc(torch.argmax(pred, dim = 1), torch.argmax(labels, dim = 1))
            self.valid_auc(pred, torch.argmax(labels, dim = 1))
        else:
            self.valid_acc(pred, labels)
            self.valid_auc(pred, labels)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        self.log('valid_auc', self.valid_auc, on_step=True, on_epoch=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params['LR']) 
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]