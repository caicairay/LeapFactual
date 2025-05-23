from pathlib import Path
import torch
import time
from torch.utils.data import TensorDataset, DataLoader
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from exp_utils import Generator, Human_Annotator, wrap_unet1d

device = 'cuda'
# Precompute
total_data_num = 20000
# StyleGAN
truncation_psi=1.
noise_mode='const'
stylegan_weight_path = 'checkpoints/stylegan3-r-ffhq-1024x1024.pkl'
# Flow Matching Model
dim = 512
# Training
sigma = 1e-4
batch_size = 32
lr = 2e-4
weight_decay = 1e-5
n_epoch = 120
start_epoch = 0
save_every = 30

generator = Generator(device = device, truncation_psi=truncation_psi, noise_mode=noise_mode, pkl_path = stylegan_weight_path)
annotator = Human_Annotator(device = device)
print('Generator and Classifier Loaded')

# Dataset
try:
    data_w = torch.load(f"checkpoints/precompute_w_{total_data_num}.pt").cpu()
    label = torch.load(f"checkpoints/precompute_y_{total_data_num}.pt").cpu()
except:
    batch_size = 64
    data_w_pos_list = []
    data_w_neg_list = []
    label_pos_list = []
    label_neg_list = []
    pos_counter = 0
    neg_counter = 0
    while (pos_counter < total_data_num // 2) or (neg_counter < total_data_num // 2):
        z, img, w = generator(batch_size)
        y = annotator(img)
        if pos_counter < total_data_num // 2:
            selection = y == 0
            data_w_pos_list.append(w[selection].detach().cpu())
            label_pos_list.append(y[selection].detach().cpu())
            pos_counter += selection.sum()
        if neg_counter < total_data_num // 2:
            selection = y == 1
            data_w_neg_list.append(w[selection].detach().cpu())
            label_neg_list.append(y[selection].detach().cpu())
            neg_counter += selection.sum()
    data_w = torch.cat(data_w_pos_list + data_w_neg_list, dim = 0)
    label = torch.cat(label_pos_list + label_neg_list, dim = 0)
    torch.save(data_w.detach().cpu(), f"checkpoints/precompute_w_{total_data_num}.pt")
    torch.save(label.detach().cpu(), f"checkpoints/precompute_y_{total_data_num}.pt")
print('Data precomputing finished')
# Create Dataloader
dataset = TensorDataset(data_w, label)
loader = DataLoader(dataset, batch_size=batch_size,pin_memory=True, shuffle=True)

# Model
model = wrap_unet1d(
        dim = dim,
        num_classes = 2,
).to(device)
print('Flow Matching Model Created')
if start_epoch > 0:
    model.load_state_dict(torch.load('checkpoints/ffhq_unet1d_w_{}_epoch{:05d}.pt'.format(total_data_num, start_epoch), weights_only=True))
    model.train()
    print("Flow Matching Model Loaded")

# Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
FM = ConditionalFlowMatcher(sigma=sigma)
start = time.time()
model.train()
Path("./checkpoints").mkdir(parents=True, exist_ok=True)
print('Entering training loop')
for epoch in range(start_epoch, n_epoch):
    print(f"Processing epoch {epoch}", end = '\r')
    for z1, y in loader:
        z1 = z1.to(device)
        y = y.to(device)
        z0 = torch.randn_like(z1)

        t, zt, ut = FM.sample_location_and_conditional_flow(z0, z1)
        vt = model(t, zt, y)

        optimizer.zero_grad()
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % save_every == 0:
        end = time.time()
        print(f"{epoch+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        torch.save(model.state_dict(), "checkpoints/ffhq_unet1d_w_{}_epoch{:05d}.pt".format(total_data_num, epoch+1))
torch.save(model.state_dict(), "checkpoints/ffhq_unet1d_w_{}_final.pt".format(total_data_num))