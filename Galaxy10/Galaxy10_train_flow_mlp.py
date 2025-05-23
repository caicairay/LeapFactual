from torch.nn import functional as F

import sys
sys.path.append("../training_scripts")
from models.classifiers import cls_models
from models.VAE import vae_models
from data.dataset_lit import DatasetLit

from tqdm import tqdm

import torch
from torchcfm.utils import *
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import time
from torchmetrics.regression import MeanSquaredError
import matplotlib.pyplot as plt

# configs
device = 'cuda'
sigma = 0.
dim = 64
batch_size = 256
n_epochs = 500
start_epoch = 0
lr = 5e-4
num_classes = 10
mlp_width = 256
milestones = [400]
save_every = 100
cls_ckpt_path = '../training_scripts/logs/Galaxy10_VGG_20/version_0/checkpoints/last.ckpt'
vae_ckpt_path = '../training_scripts/logs/Galaxy10_VAE/version_0/checkpoints/last.ckpt'

# build dataset
dataset = DatasetLit(name = 'Galaxy10', image_size = 128, 
                     train_batch_size = batch_size, val_batch_size = 128, test_batch_size = 128,
                     num_workers = 8, pin_memory = True, download = False,
                     train_root= './galaxy10_data/image_data_train_20/',
                     test_root= './galaxy10_data/image_data_test/'
                     )
dataset.setup()

# Build the classifier
classifier = cls_models['VGG'](num_classes, bce=False).to(device)
# Load the weights:
classifier.load_checkpoint(cls_ckpt_path)
classifier.eval()
for param in classifier.parameters():
    param.require_grad = False

# Build VAE
vae = vae_models['WAE_MMD'](in_channels=3, image_size=128,total_latent_dim=64,num_classes=0,img_size=128,hidden_dims=[16, 32, 64, 128, 256],reg_weight=10., kernel_type='imq', latent_var = 2.).to(device)
# Load the weights
vae.load_checkpoint(vae_ckpt_path)
vae.eval()
for param in vae.parameters():
    param.require_grad = False

# Build and train flow
class MLP_CFM(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, cond=False, cond_dim = None):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        if cond and cond_dim is None:
            cond_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0) + (cond_dim if cond else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, t, x):
        while t.dim() > 1:
            t = t[:, 0]
        if t.dim() == 0:
            t = t[:, None]
        if t.dim() == 1 and t.shape[0] == 1:
            t = t.repeat(x.shape[0])
        return self.net(torch.cat([x, t[:, None]], dim = 1))

model_cfm = MLP_CFM(dim=dim, time_varying=True, cond=True, cond_dim = num_classes, w=mlp_width).to(device)
optimizer = torch.optim.Adam(model_cfm.parameters(), lr = lr)
lr_sche = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
if start_epoch > 0:
    checkpoint = torch.load('checkpoints/galaxy10_flow_weights_mlp_epoch{:05d}.pt'.format(start_epoch))
    model_cfm.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_sche.load_state_dict(checkpoint['lr_sche'])
model_cfm.train()
FM = ConditionalFlowMatcher(sigma=sigma)
metric_mse = MeanSquaredError().to(device)
mse_record = []
pbar = tqdm(range(start_epoch, n_epochs))
start = time.time()
for epoch in pbar:
    metric_mse.reset()
    for img, label in dataset.train_dataloader():
        img = img.to(device)
        x1 = vae.encode(img)
        y = F.one_hot(torch.argmax(classifier(img), dim = 1), num_classes).to(torch.float)
        x0 = torch.randn_like(x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = model_cfm(t, torch.cat([xt, y], dim=-1))
        metric_mse.update(vt, ut)
        
        optimizer.zero_grad()
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()
    lr_sche.step()
    mse = metric_mse.compute()
    pbar.set_description("Epoch {:03d}: Loss = {:.3f}".format(epoch + 1, mse))
    mse_record.append(mse)
    if (epoch + 1) % save_every == 0:
        end = time.time()
        print(f"{epoch+1}: loss {mse:0.3f} time {(end - start):0.2f}")
        start = end
        checkpoint = { 
            'epoch': epoch,
            'model': model_cfm.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sche': lr_sche.state_dict()}
        torch.save(model_cfm.state_dict(), "checkpoints/galaxy10_flow_weights_mlp_epoch{:05d}.pt".format(epoch+1))

checkpoint = { 
    'epoch': epoch,
    'model': model_cfm.state_dict(),
    'optimizer': optimizer.state_dict(),
    'lr_sche': lr_sche.state_dict()}
torch.save(model_cfm.state_dict(), 'checkpoints/galaxy10_flow_weights_mlp.pt')
mse_record_numpy = torch.hstack(mse_record).cpu().numpy()
plt.plot(mse_record_numpy)
plt.savefig('galaxy10_flow_mlp_training_loss.png')