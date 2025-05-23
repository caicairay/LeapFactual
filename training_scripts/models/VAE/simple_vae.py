import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 num_classes,
                 hidden_dims = None,
                 **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        modules = []
        if hidden_dims is None:
            hidden_dims = [8,16,32,32]
        # Input layers
        self.input_layer = nn.ZeroPad2d(2)
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(negative_slope=0.2))
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*4 + num_classes, latent_dim * 2)
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(negative_slope=0.2))
            )
        self.decoder = nn.Sequential(*modules)
        # output layers
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(negative_slope=0.2),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 5, padding= 0),
                            nn.Sigmoid())

    def encode(self, x, cond = None):
        result = self.input_layer(x)
        result = self.encoder(result)
        result = torch.flatten(result, start_dim=1)
        result = result if cond is None else torch.cat([result, cond], dim = 1)
        mu, log_var = self.fc(result).chunk(2, dim = 1)
        return [mu, log_var]
    def decode(self, z, cond = None):
        input = z if cond is None else torch.cat([z, cond], dim=1)
        result = self.decoder_input(input).view(-1, 32, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, cond = None):
        mu, log_var = self.encode(input, cond)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z, cond), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['kld_weight']
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples,
               device, **kwargs):
        z = torch.randn(num_samples,
                        self.latent_dim).to(device)
        if self.num_classes > 0:
            if num_samples <= 10:
                cond = torch.arange(10).to(device)
            else: 
                cond = torch.randint(low=0, high=self.num_classes, size=(z.size(0),)).to(device)
            cond = F.one_hot(cond, self.num_classes)
        else:
            cond = None
        samples = self.decode(z, cond).to(device)
        return samples

    def generate(self, x, cond = None):
        mu, log_var = self.encode(x, cond)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, cond)