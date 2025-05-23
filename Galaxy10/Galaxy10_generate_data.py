import os
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F
from models.classifiers import cls_models
from models.generative_models.VAE import vae_models
from data.dataset_lit import DatasetLit
from torchdyn.core import NeuralODE
from torchcfm.utils import *

# configs
device = 'cuda'
cls_ckpt_path = '../training_scripts/logs/Galaxy10_VGG_20/version_0/checkpoints/last.ckpt'
vae_ckpt_path = '../training_scripts/logs/Galaxy10_VAE/version_0/checkpoints/last.ckpt'
flow_ckpt_path = './checkpoints/galaxy10_flow_weights_mlp.pt'
save_data_root = './galaxy10_data/synthetic_data'
dim = 64
batch_size = 512
# LeapFactual
n_steps = 15
gamma_b = 0.1
N_b = 250
gamma_i_lift = 1.0
gamma_i_land = 1.025
N_i = 50

# build dataset
dataset = DatasetLit(name = 'Galaxy10', image_size = 128, 
                     train_batch_size = batch_size, val_batch_size = 128, test_batch_size = 128,
                     num_workers = 8, pin_memory = True, download = False,
                     train_root= './galaxy10_data/image_data_train_20/',
                     test_root= './galaxy10_data/image_data_test/'
                     )
dataset.setup()
# Build and load classifier
classifier = cls_models['VGG'](10, bce=False).to(device)
classifier.load_checkpoint(cls_ckpt_path)
classifier.eval()

# Build and load VAE
vae = vae_models['WAE_MMD'](in_channels=3, image_size=128,total_latent_dim=64,num_classes=0,img_size=128,hidden_dims=[16, 32, 64, 128, 256],reg_weight=10., kernel_type='imq', latent_var = 2.).to(device)
vae.load_checkpoint(vae_ckpt_path)
vae.eval()

# ## Build and load Flows
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
model_cfm = MLP_CFM(dim=dim, time_varying=True, cond=True, cond_dim = 10, w=256).to(device)
checkpoint = torch.load(flow_ckpt_path)
model_cfm.load_state_dict(checkpoint)
model_cfm.eval()
class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model, c, step_size= 1.):
        super().__init__()
        self.model = model
        self.c = c
        self.step_size = step_size

    def forward(self, t, x, *args, **kwargs):
        if t.dim() == 0: t = t.unsqueeze(0)
        return self.step_size * self.model(t, torch.cat([x, self.c], dim=-1))
        
@torch.no_grad()
def leap(model, source_points, source_y, target_y, gamma_lift = 1., gamma_land = 1., n_steps = 10, t_lower = 0., t_upper = 1.):
    if gamma_lift > 0.:
        node_lift = NeuralODE(torch_wrapper(model, source_y, step_size = gamma_lift), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        # lifting transport
        traj_lift = node_lift.trajectory(
            source_points,
            t_span=torch.linspace(t_upper, t_lower, n_steps),
        )
    land_start_points = traj_lift[-1].detach().clone() if gamma_lift > 0. else source_points
    node_land = NeuralODE(torch_wrapper(model, target_y, step_size = gamma_land), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    # landing transport
    traj_land = node_land.trajectory(
            land_start_points,
            t_span=torch.linspace(t_lower, t_upper, n_steps),
        )
    return traj_land

@torch.no_grad()
def progressive_leap(model, source_points, classifier, target_y, gamma_lift = 0.1, gamma_land = 0.1, max_itr = 10, n_steps = 10, t_lower = 0., t_upper = 1., mute = False):
    traj_list = []
    lift_start_points = source_points
    for itr in range(max_itr):
        source_y = classifier(lift_start_points)
        if not mute: print('Processing itration {:03d}, predicted label {}'.format(itr, torch.argmax(source_y, dim = 1 ).detach().cpu().numpy()), end = '\r')
        traj_land = leap(model, lift_start_points, source_y, target_y,
                         gamma_lift = gamma_lift, gamma_land = gamma_land,
                         n_steps = n_steps, t_lower = t_lower, t_upper=t_upper)
        lift_start_points = traj_land[-1]
        traj_list.append(lift_start_points)
    return traj_list

@torch.no_grad()
def generator(x):
    return vae.decode(x, cond = None)
@torch.no_grad()
def classifier_traj(x): 
    return F.one_hot(torch.argmax(classifier(generator(x)), dim = 1), 10).to(torch.float)

outer_counter = 0
with torch.no_grad():
    for train_img, train_label in dataset.train_dataloader():
        train_img = train_img.to(device)
        train_label_onehot = F.one_hot(train_label, 10).to(device = device, dtype = torch.float)
        source_points = vae.encode(train_img)
        # recon_img = generator(source_points)
        # recon_img_numpy = recon_img.permute(0, 2, 3, 1).detach().cpu().numpy()
        # roll the gt label 9 times to generate CEs for all the rest labels
        for num_shift in range(0, 10):
            if num_shift == 0:
                target_y = train_label
                cf_imgs = generator(source_points)
                cf_robust_imgs = cf_imgs
            else:
                target_y_onehot = train_label_onehot.roll(shifts = num_shift, dims = 1)
                target_y = torch.argmax(target_y_onehot, dim = 1)
                # generate CF
                traj_list = progressive_leap(model_cfm, source_points, classifier_traj, target_y_onehot,
                                gamma_lift = gamma_b, gamma_land = gamma_b,
                                max_itr = N_b, n_steps = n_steps, mute= True
                                )
                cf_imgs = generator(traj_list[-1])
                # generate robust CF
                source_points_robust = traj_list[-1]
                traj_list_robust = progressive_leap(model_cfm, source_points_robust, classifier_traj, target_y_onehot,
                                    gamma_lift = gamma_i_lift, gamma_land = gamma_i_land,
                                    max_itr = N_i, n_steps = n_steps, mute= True
                                )
                cf_robust_imgs = generator(traj_list_robust[-1])
            # save data
            cf_imgs_numpy = cf_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            cf_robust_imgs_numpy = cf_robust_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            target_y_numpy = target_y.detach().cpu().numpy()
            inner_counter = outer_counter
            for cf_img, cf_robust_img, label in zip(cf_imgs_numpy, cf_robust_imgs_numpy, target_y_numpy):
                # save CE
                save_img = Image.fromarray((cf_img * 255).astype(np.uint8))
                save_img.save(os.path.join(save_data_root, f'CE_{N_i}', f'ans_{label}', f'img_{inner_counter:05d}_{num_shift}.jpeg'))
                # save robust CE
                save_img = Image.fromarray((cf_robust_img * 255).astype(np.uint8))
                save_img.save(os.path.join(save_data_root, f'robustCE_{N_i}', f'ans_{label}', f'img_{inner_counter:05d}_{num_shift}.jpeg'))
                inner_counter += 1
        outer_counter += batch_size