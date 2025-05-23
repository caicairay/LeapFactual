import torch
import matplotlib.pyplot as plt

from torcheval.metrics import BinaryAccuracy, BinaryAUROC
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from exp_utils import Generator, Human_Annotator, wrap_unet1d, progressive_leap

device = 'cuda'
# StyleGAN
truncation_psi=1.
noise_mode='const'
stylegan_weight_path = 'checkpoints/stylegan3-r-ffhq-1024x1024.pkl'
# Unet
unet_dim = 512
unet_weight_path = 'checkpoints/ffhq_unet1d_w_20000_final.pt'
# Quantitative evaluation
num_img = 1024
batch_size = 32
gamma_lift = .8
gamma_land = .8
n_steps = 15 
max_itr = 5

# Load Models
generator = Generator(device = device, truncation_psi=truncation_psi, noise_mode=noise_mode, pkl_path = stylegan_weight_path)
annotator = Human_Annotator(device = device)
classifier = lambda z: annotator(generator.generate(z))
model = wrap_unet1d(
        dim = unet_dim,
        num_classes = 2,
).to(device)
model.load_state_dict(torch.load(unet_weight_path, weights_only=True))#,  map_location= {'cuda:5':device}))
model.eval()

# Create Metrics
metric_acc_list = [BinaryAccuracy(),BinaryAccuracy(),BinaryAccuracy(),BinaryAccuracy()]
metric_auc_list = [BinaryAUROC(),BinaryAUROC(),BinaryAUROC(),BinaryAUROC()]
metric_ssim_list = [StructuralSimilarityIndexMeasure(data_range = (-1, 1)).to(device),
                    StructuralSimilarityIndexMeasure(data_range = (-1, 1)).to(device),
                    StructuralSimilarityIndexMeasure(data_range = (-1, 1)).to(device),
                    StructuralSimilarityIndexMeasure(data_range = (-1, 1)).to(device)]
metric_psnr_list = [PeakSignalNoiseRatio(data_range = (-1, 1)).to(device),
                    PeakSignalNoiseRatio(data_range = (-1, 1)).to(device),
                    PeakSignalNoiseRatio(data_range = (-1, 1)).to(device),
                    PeakSignalNoiseRatio(data_range = (-1, 1)).to(device)]
metric_lpip_list = [LearnedPerceptualImagePatchSimilarity(net_type = 'squeeze').to(device), 
                LearnedPerceptualImagePatchSimilarity(net_type = 'squeeze').to(device),
                LearnedPerceptualImagePatchSimilarity(net_type = 'squeeze').to(device),
                LearnedPerceptualImagePatchSimilarity(net_type = 'squeeze').to(device)]

# Evaluation Loop
counter = 0
while counter < num_img:
    # Randomly sample z
    z = torch.randn(batch_size, 512, device=device)
    # Convert to w-space
    source_data = generator.G.mapping(z, c = None)[:, 0]
    source_imgs = generator.generate(source_data)
    source_y = annotator(source_imgs)
    # Evaluation
    target_y = 1 - source_y
    source_data_itr = source_data
    for i in range(len(metric_acc_list)):
        traj_list = progressive_leap(model, source_data_itr, classifier, target_y, gamma_lift = gamma_lift, gamma_land = gamma_land, n_steps = n_steps, max_itr = max_itr)
        cf_imgs = generator.generate(traj_list[-1])
        # Correctness
        pred_y = annotator(cf_imgs)
        metric_acc_list[i].update(pred_y, target_y)
        metric_auc_list[i].update(pred_y, target_y)
        # Similarity
        metric_ssim_list[i].update(cf_imgs.clamp(-1, 1), source_imgs.clamp(-1, 1))
        metric_psnr_list[i].update(cf_imgs.clamp(-1, 1), source_imgs.clamp(-1, 1))
        metric_lpip_list[i].update(cf_imgs.clamp(-1, 1), source_imgs.clamp(-1, 1))
        source_data_itr = traj_list[-1]
    # Ending
    counter += batch_size
# Saving Files
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
axs[0].plot([5, 10, 15, 20], [metric_acc.compute() for metric_acc in metric_acc_list])
axs[1].plot([5, 10, 15, 20], [metric_auc.compute() for metric_auc in metric_auc_list])
axs[2].plot([5, 10, 15, 20], [metric_ssim.compute().cpu().numpy() for metric_ssim in metric_ssim_list])
axs[3].plot([5, 10, 15, 20], [metric_psnr.compute().cpu().numpy() for metric_psnr in metric_psnr_list])
axs[4].plot([5, 10, 15, 20], [metric_lpip.compute().cpu().numpy() for metric_lpip in metric_lpip_list])
fig.savefig('FFHQ_metrics_history.png', dpi = 300)

with open("FFHQ_metrics_history.txt", "a") as f:
    f.write("Accuracy:" + ",".join([str(metric_acc.compute().cpu().numpy()) for metric_acc in metric_acc_list]) + "\n")
    f.write("AUC:" + ",".join([str(metric_auc.compute().cpu().numpy()) for metric_auc in metric_auc_list])+ "\n")
    f.write("SSIM:" + ",".join([str(metric_ssim.compute().cpu().numpy()) for metric_ssim in metric_ssim_list])+ "\n")
    f.write("PSNR" + ",".join([str(metric_psnr.compute().cpu().numpy()) for metric_psnr in metric_psnr_list])+ "\n")
    f.write("LPIP" + ",".join([str(metric_lpip.compute().cpu().numpy()) for metric_lpip in metric_lpip_list])+ "\n")