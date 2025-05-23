import torch
from exp_utils import Generator
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

device = 'cuda'
# StyleGAN
truncation_psi=1.
noise_mode='const'
stylegan_weight_path = 'checkpoints/stylegan3-r-ffhq-1024x1024.pkl'
# Unet
unet_dim = 512
unet_weight_path = 'checkpoints/ffhq_unet1d_argmax_w_20000_final.pt'
# ## Quantitative evaluation
num_img = 1024
batch_size = 32
# selection_random_seed = 2023

generator = Generator(device = device, truncation_psi=truncation_psi, noise_mode=noise_mode, pkl_path = stylegan_weight_path)

metric_ssim = StructuralSimilarityIndexMeasure(data_range = (-1, 1)).to(device)
metric_psnr = PeakSignalNoiseRatio(data_range = (-1, 1)).to(device)
metric_lpip = LearnedPerceptualImagePatchSimilarity(net_type = 'squeeze').to(device)
counter = 0
while counter < num_img:
    z = torch.randn(batch_size, 512, device=device)
    source_data = generator.G.mapping(z, c = None)[:, 0]
    source_imgs = generator.generate(source_data)
    z = torch.randn(batch_size, 512, device=device)
    target_data = generator.G.mapping(z, c = None)[:, 0]
    target_imgs = generator.generate(target_data)
    # evaluation
    metric_ssim.update(target_imgs.clamp(-1, 1), source_imgs.clamp(-1, 1))
    metric_psnr.update(target_imgs.clamp(-1, 1), source_imgs.clamp(-1, 1))
    metric_lpip.update(target_imgs.clamp(-1, 1), source_imgs.clamp(-1, 1))
    # ending
    counter += batch_size
with open("FFHQ_metrics_ref.txt", "a") as f:
    f.write("SSIM:" + str(metric_ssim.compute().cpu().numpy())+ "\n")
    f.write("PSNR" +  str(metric_psnr.compute().cpu().numpy())+ "\n")
    f.write("LPIP" +  str(metric_lpip.compute().cpu().numpy())+ "\n")