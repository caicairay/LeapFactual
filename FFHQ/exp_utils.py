import torch
import pickle
import clip
import sys
sys.path.append('../external_pkgs/stylegan3')
from PIL import Image
import torch.nn as nn

from torchdyn.core import NeuralODE
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from torchcfm.conditional_flow_matching import *
from denoising_diffusion_pytorch import KarrasUnet1D

class Human_Annotator(nn.Module):
    def __init__(self, text_choices = ["smiling face", "face"], device = 'cpu', model= "ViT-B/32"):
        super().__init__()
        self.model, clip_preprocess = clip.load(model, device=device)
        self.text = clip.tokenize(text_choices).to(device)
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor
    def forward(self, img):
        image = self.preprocess(img.clamp(-1, 1))
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, self.text)
            probs = logits_per_image.softmax(dim=-1)
        return torch.argmax(probs, dim = 1)

class Generator(nn.Module):
    def __init__(self, pkl_path = 'checkpoints/stylegan3-r-ffhq-1024x1024.pkl', device = 'cpu', truncation_psi=1., noise_mode='none'):
        super().__init__()
        self.device = device
        with open(pkl_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
    def forward(self, n_samples):
        with torch.no_grad():
            z = torch.randn([n_samples, self.G.z_dim]).to(self.device)    # latent codes
            c = None                                # class labels (not used in this example)
            img = self.G(z, c, noise_mode = self.noise_mode)                           # NCHW, float32, dynamic range [-1, +1], no truncation
            w = self.G.mapping(z, c)
            return z, img, w[:, 0, :]
    def to_pil(self, img):
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_img = Image.fromarray(img.cpu().numpy(), 'RGB')
        return pil_img
    def generate(self, w, with_grad = False):
        w = w.unsqueeze(1).repeat(1, 16, 1)
        # print(w.shape)
        if with_grad:
            img = self.G.synthesis(w, noise_mode=self.noise_mode)#, truncation_psi=self.truncation_psi)                           # NCHW, float32, dynamic range [-1, +1], no truncation
        with torch.no_grad():       
            img = self.G.synthesis(w, noise_mode=self.noise_mode)#, truncation_psi=self.truncation_psi)                           # NCHW, float32, dynamic range [-1, +1], no truncation
        return img

class wrap_unet1d(KarrasUnet1D):
    def __init__(self, dim, num_classes):
        super().__init__(seq_len = dim, num_classes=num_classes, channels=1)
    def forward(self, t, x, y):
        u = super().forward(x.unsqueeze(1), t, class_labels = y).squeeze(1)
        return u
        
class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model, c, step_size= 1.):
        super().__init__()
        self.model = model
        self.c = c
        self.step_size = step_size

    def forward(self, t, x, *args, **kwargs):
        if t.dim() == 0: t = t.unsqueeze(0)
        return self.step_size * self.model(t, x, self.c)

def make_plot(traj_list, generator, num_img, ax):
    plot_list = []
    for w in traj_list:
        img = generator.generate(w)
        img =  (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        plot_list.append(img)
    grid = make_grid(
        torch.cat(plot_list, dim = 0), value_range=(0, 255), padding=0, nrow=num_img
    )
    img = ToPILImage()(grid)
    ax.imshow(img)
    ax.set_axis_off()

@torch.no_grad()
def leap(model, source_points, source_y, target_y, gamma_lift = 1., gamma_land = 1.,  n_steps = 10, t_lower = 0., t_upper = 1.):
    if gamma_lift > 0:
        node_lift = NeuralODE(torch_wrapper(model, source_y, gamma_lift), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        # lifting transport
        traj_lift = node_lift.trajectory(
            source_points,
            t_span=torch.linspace(t_upper, t_lower, n_steps),
        )
    land_start_points = traj_lift[-1].detach().clone() if gamma_lift > 0 else source_points
    node_land = NeuralODE(torch_wrapper(model, target_y, gamma_land), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    # landing transport
    traj_land = node_land.trajectory(
            land_start_points,
            t_span=torch.linspace(t_lower, t_upper, n_steps),
        )
    return traj_land

@torch.no_grad()
def progressive_leap(model, source_points, classifier, target_y, gamma_lift = 0.1, gamma_land = 0.1, max_itr = 10, n_steps = 10, t_lower = 0., t_upper = 1.):
    traj_list = []
    lift_start_points = source_points.detach().clone()
    for itr in range(max_itr):
        source_y = classifier(lift_start_points)
        print(f'Processing itration {itr}, predicted label {source_y.detach().cpu().numpy()}')
        traj_land = leap(model, lift_start_points, source_y, target_y, gamma_lift = gamma_lift, gamma_land = gamma_land, 
                            n_steps = n_steps, t_lower = t_lower, t_upper=t_upper)
        lift_start_points = traj_land[-1].detach().clone()
        traj_list.append(lift_start_points)
    print(f'Finish, predicted label {classifier(lift_start_points).detach().cpu().numpy()}')
    return traj_list