"""
Adapted from https://github.com/LukeDitria/CNN-VAE/blob/master/RES_VAE_Dynamic.py
"""
import torch
from torch import nn
from torch.nn import functional as F
from .losses import GANLoss, NLayerDiscriminator, VGGPerceptualLoss_Galaxy

def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        ValueError("norm_type must be bn or gn")

class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3,  norm_type="bn"):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, max(channel_out // 2, 1) + channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(max(channel_out // 2, 1), norm_type=norm_type)

        self.conv2 = nn.Conv2d(max(channel_out // 2, 1), channel_out, kernel_size, 1, kernel_size // 2)

        self.act_fnc = nn.LeakyReLU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, max(channel_in // 2, 1) + channel_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(max(channel_in // 2, 1), norm_type=norm_type)

        self.conv2 = nn.Conv2d(max(channel_in // 2, 1), channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.LeakyReLU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))
        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class ResBlock(nn.Module):
    """
    Residual block
    """
    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        first_out = max(channel_in // 2, 1) if channel_in == channel_out else max(channel_in // 2, 1) + channel_out
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(max(channel_in // 2, 1), norm_type=norm_type)
        self.conv2 = nn.Conv2d(max(channel_in // 2, 1), channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.LeakyReLU()
        self.skip = channel_in == channel_out
        self.bttl_nk = max(channel_in // 2, 1)

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))
        x_cat = self.conv1(x)
        x = x_cat[:, :self.bttl_nk]
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip

class WAE_MMD(nn.Module):
    def __init__(self,
                 in_channels,
                 total_latent_dim,
                 num_classes,
                 img_size,
                 hidden_dims = [16, 32, 64, 32],
                 reg_weight= 100,
                 kernel_type='imq',
                 latent_var = 2.,
                 **kwargs):
        super(WAE_MMD, self).__init__()

        self.num_classes = num_classes
        self.total_latent_dim = total_latent_dim
        self.img_size = img_size
        self.num_layers = len(hidden_dims)
        self.embedding_shape = [hidden_dims[-1], img_size // (2 ** self.num_layers), img_size // (2 ** self.num_layers)]
        self.embedding_length = torch.prod(torch.tensor(self.embedding_shape))
        self.num_input_dim = in_channels * img_size ** 2
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var
        self.perceptual_loss = VGGPerceptualLoss_Galaxy()
        self.recon_discriminator = NLayerDiscriminator(input_nc=in_channels, n_layers=3)
        self.gan_loss = GANLoss(gan_mode='lsgan')

        # Build Encoder
        local_in_channels = in_channels
        modules = []
        for h_dim in hidden_dims:
            modules.append(ResBlock(local_in_channels, local_in_channels))
            modules.append(
                ResDown(local_in_channels, h_dim)
            )
            local_in_channels = h_dim
        modules.append(ResBlock(local_in_channels, local_in_channels))
        self.encoder = nn.Sequential(*modules)
        # Build the bottleneck layers
        self.fc_in = nn.Sequential(
            nn.Linear(self.embedding_length, 512),
            nn.Linear(512, total_latent_dim)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(total_latent_dim, 512), 
            nn.Linear(512, self.embedding_length), 
        )
        hidden_dims.reverse()
        # CVAE only
        if self.num_classes > 0:
            self.fc_cond = nn.Sequential(
                nn.Linear(self.num_classes, 512), 
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 512), 
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 512), 
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, self.embedding_length), 
            )
            # Build Condition Embed
            self.cond_embed = nn.ModuleList()
            for i in range(len(hidden_dims)):
                self.cond_embed.append(
                    nn.Sequential(
                        ResBlock(hidden_dims[i], hidden_dims[i]),
                        ResUp(hidden_dims[i], 
                            channel_out=hidden_dims[i + 1] if (i + 1) < len(hidden_dims) else 8, 
                            )
                ))
            self.z_discriminator = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.total_latent_dim, 512), 
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512), 
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512), 
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        # Build Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.decoder.append(
                nn.Sequential(
                    ResBlock(hidden_dims[i] * (2 if num_classes > 0 else 1), hidden_dims[i]),
                    ResUp(hidden_dims[i], channel_out=hidden_dims[i + 1] if (i + 1) < len(hidden_dims) else 8),
                    )
            )
        hidden_dims.reverse()
        # Build output layer
        self.output = nn.Sequential(
                ResBlock(8, 8),
                nn.Conv2d(in_channels= 8, out_channels= in_channels,
                        kernel_size= 5, stride= 1, padding= 2
                        ),
                nn.Sigmoid())
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = torch.flatten(self.encoder(input), start_dim=1)
        return self.fc_in(result)

    def decode(self, z, cond = None):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        if cond is not None:
            cond_middle = self.fc_cond(cond).view(-1, *self.embedding_shape)
            decoder_middle = self.fc_out(z).view(-1, *self.embedding_shape)
            for l_cond, l_dec in zip(self.cond_embed, self.decoder):
                decoder_input = torch.cat([decoder_middle, cond_middle], dim = 1)
                decoder_middle = l_dec(decoder_input)
                cond_middle = l_cond(cond_middle)
            return self.output(decoder_middle)
        else:
            result = self.fc_out(z).view(-1, *self.embedding_shape)
            for layer in self.decoder:
                result = layer(result)
            return self.output(result)

    def forward(self, input, **kwargs):
        labels = kwargs['labels']
        labels = F.one_hot(labels.squeeze(), self.num_classes).to(torch.float) if self.num_classes > 0 else None
        z = self.encode(input)
        recon_img = self.decode(z, labels)
        return [recon_img, z]
    
    def forward_gan(self, input, **kwargs):
        labels = kwargs['labels']
        labels = F.one_hot(labels.squeeze(), self.num_classes).to(torch.float) if self.num_classes > 0 else None
        with torch.no_grad():
            z = self.encode(input)
            recon_img = self.decode(z, labels)
        if labels is not None:
            return self.recon_discriminator(recon_img.detach()), self.z_discriminator(z.detach())
        else:
            return self.recon_discriminator(recon_img.detach()), None

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        real_img = kwargs['real_img']
        y_true = kwargs['labels']
        y_true = F.one_hot(y_true.squeeze(), self.num_classes).to(torch.float)  if self.num_classes > 0 else None
        perceptual_weight = kwargs['M_P']
        recon_weight = kwargs['M_R']
        gan_weight = kwargs['M_G']
        adv_weight = kwargs['M_A']

        batch_size = real_img.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        recon_img = args[0]
        z = args[1] 
        
        losses = {}
        if recon_weight > 0:
            recons_loss = recon_weight * F.mse_loss(recon_img, real_img)
            losses['Reconstruction_Loss'] = recons_loss
        if perceptual_weight > 0:
            perceptual_loss = perceptual_weight * self.perceptual_loss(recon_img, real_img)
            losses['Perceptual_Loss'] = perceptual_loss
        if reg_weight > 0:
            mmd_loss = self.compute_mmd(z, reg_weight)
            losses['MMD_Loss'] = mmd_loss
        if gan_weight > 0:
            gan_loss = gan_weight * self.gan_loss(self.recon_discriminator(recon_img), target_is_real=True)
            losses['GAN_Loss'] = gan_loss
        if adv_weight > 0 and y_true is not None:
            adv_loss = adv_weight * F.cross_entropy(self.z_discriminator(z), F.one_hot(torch.randint(low = 0, high = 10, size = (z.shape[0], )), self.num_classes).to(dtype= torch.float, device = z.device))
            losses['ADV_Loss'] = adv_loss
        
        loss = torch.tensor(0.).to(real_img.device)
        for l in losses.values():
            loss += l
        losses['loss'] = loss
        return losses
        
    def loss_adv(self,
                *args,
                **kwargs):
        recon_pred = args[0]
        z_pred = args[1]
        y_true = kwargs['labels']
        y_true = F.one_hot(y_true.squeeze(), self.num_classes).to(torch.float)  if self.num_classes > 0 else None

        real_img = kwargs['real_img']
        recon_fake_loss = self.gan_loss(recon_pred, target_is_real=False) 
        recon_real_loss = self.gan_loss(self.recon_discriminator(real_img), target_is_real=True) 
        recon_adv_loss = (recon_real_loss + recon_fake_loss) / 2
        if z_pred is not None:
            z_adv_loss = F.cross_entropy(z_pred, y_true)
            D_loss = z_adv_loss + recon_adv_loss
            return {'D_loss': D_loss, 'Z_D_loss': z_adv_loss, 'RECON_D_loss': recon_adv_loss}
        else:
            D_loss = recon_adv_loss
            return {'D_loss': D_loss}

    def sample(self,
               num_samples,
               current_device, 
               return_latent = False,
               **kwargs):
        z = torch.randn(num_samples,
                        self.total_latent_dim)
        z = z.to(current_device)
        if self.num_classes > 0:
            labels = torch.randint(low=0, high = self.num_classes, size= (num_samples, ))
            labels = F.one_hot(labels.squeeze(), self.num_classes).to(dtype = torch.float, device = current_device) if self.num_classes > 0 else None
        else:
            labels = None
        samples = self.decode(z, labels)
        if return_latent:
            return samples, z
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, **kwargs)[0]
    
    def load_checkpoint(self, root, map_location = None):
        new_weights = self.state_dict()
        old_weights = list(torch.load(root, map_location=map_location)['state_dict'].items())
        i=0
        for k, _ in new_weights.items():
            new_weights[k] = old_weights[i][1]
            i += 1
        self.load_state_dict(new_weights)
    
    def calc_kl_loss(self, mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

    def compute_kernel(self,
                       x1,
                       x2):
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result

    def compute_rbf(self,
                    x1,
                    x2,
                    eps = 1e-7):
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var
        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,
                               x1,
                               x2,
                               eps = 1e-7):
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))
        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()
        return result

    def compute_mmd(self, z, reg_weight):
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = reg_weight * prior_z__kernel.mean() + \
              reg_weight * z__kernel.mean() - \
              2 * reg_weight * priorz_z__kernel.mean()
        return mmd