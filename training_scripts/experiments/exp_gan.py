from torch import optim
import pytorch_lightning as pl
from torch.nn import functional as F
import os
import torch
import torchvision.utils as vutils
from lightning.pytorch.utilities import rank_zero_only
from .utils import frange_cycle_linear
import torchmetrics

class ExpGAN(pl.LightningModule):
    def __init__(self,
                 vae_model,
                 params: dict,
                 cls_model = None,
                 ) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.model = vae_model
        self.classifier = cls_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.train_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=(0., 1.)) 
        self.valid_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=(0., 1.)) 
        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=(0., 1.)) 
        self.valid_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=(0., 1.)) 
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):#, optimizer_idx = 0):
        real_img, labels = batch
        if self.classifier is not None:
            labels = torch.argmax(self.classifier(real_img), dim = 1).to(torch.int64)
        self.curr_device = real_img.device
        optimizer, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer)
        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              real_img = real_img,
                                              labels = labels,
                                              M_G = self.GAN_annealing[self.current_epoch],
                                              M_A = self.ADV_annealing[self.current_epoch],
                                              M_P = self.params['perceptual_weight'],
                                              M_R = self.params['recon_weight'],
                                              batch_idx = batch_idx)
        optimizer.zero_grad()
        self.manual_backward(train_loss['loss'])
        self.clip_gradients(optimizer, gradient_clip_val=self.params['gradient_clip_val'], gradient_clip_algorithm="norm")
        optimizer.step()
        self.untoggle_optimizer(optimizer)

        self.toggle_optimizer(optimizer_d)
        recon_pred, z_pred = self.model.forward_gan(real_img, labels = labels)
        train_loss_gan = self.model.loss_adv(recon_pred,
                                            z_pred,
                                            real_img = real_img,
                                            labels = labels,
                                            batch_idx = batch_idx)
        optimizer_d.zero_grad()
        self.manual_backward(train_loss_gan['D_loss'])
        self.clip_gradients(optimizer_d, gradient_clip_val=self.params['gradient_clip_val'], gradient_clip_algorithm="norm")
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)
        batch_ssim = self.train_ssim(results[0], real_img)
        batch_psnr = self.train_psnr(results[0], real_img)

        # Record the losses
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.log_dict({key: val.item() for key, val in train_loss_gan.items()}, sync_dist=True)
        self.log('train_ssim_step', batch_ssim)
        self.log('train_psnr_step', batch_psnr)
        
    def on_train_epoch_end(self):
        self.train_ssim.reset()
        self.train_psnr.reset()
        sl, sl_d = self.lr_schedulers()
        sl.step()
        sl_d.step()

    def validation_step(self, batch, batch_idx):#, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
        if self.classifier is not None:
            labels = torch.argmax(self.classifier(real_img), dim = 1).to(torch.int64)

        results = self.forward(real_img, labels= labels)
        val_loss = self.model.loss_function(*results,
                                              real_img = real_img,
                                              labels = labels,
                                              M_P = self.params['perceptual_weight'],
                                              M_R = self.params['recon_weight'],
                                              M_G = self.GAN_annealing[self.current_epoch],
                                              M_A = self.ADV_annealing[self.current_epoch],
                                              batch_idx = batch_idx)
        self.valid_ssim.update(results[0], real_img)
        self.valid_psnr.update(results[0], real_img)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    def on_validation_epoch_end(self) -> None:
        self.log_dict({"val_ssim": self.valid_ssim.compute(), "val_psnr": self.valid_psnr.compute()}, sync_dist=True)
        self.valid_ssim.reset()
        self.valid_psnr.reset()

    @rank_zero_only
    def on_validation_end(self) -> None:
        self.sample_images()
    
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input[:10].to(self.curr_device)
        if self.classifier is not None:
            labels = torch.argmax(self.classifier(test_input), dim = 1).to(torch.int64)
        else:
            labels = None

        recons_img = self.model.generate(test_input, labels = labels)
        forsave = torch.cat([test_input, 
                             recons_img,
                             ]
                             , dim = 0)
        vutils.save_image(forsave.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=10)
        sample_img = self.model.sample(10, test_input.device)
        vutils.save_image(sample_img.data,
                          os.path.join(self.logger.log_dir , 
                                       "Sample", 
                                       f"sample_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=10)
        
    def configure_optimizers(self):
        self.ADV_annealing = frange_cycle_linear(0, self.params['adv_weight'], self.trainer.max_epochs, n_cycle=1, ratio=0.1)
        self.GAN_annealing = frange_cycle_linear(0, self.params['gan_weight'], self.trainer.max_epochs, n_cycle=1, ratio=0.3)
        optims = []
        scheds = []
        descriminator_params = [] 
        other_params = []
        for name, param in self.model.named_parameters():
            if 'discriminator' in name:
                print(name)
                descriminator_params.append(param)
            else:
                other_params.append(param)
        optimizer = optim.AdamW(other_params,
                       lr=self.params['LR'],
                       weight_decay=self.params['weight_decay'])
        optimizer2 = optim.AdamW(descriminator_params,
                                lr=self.params['LR_2'])
        optims.append(optimizer)
        optims.append(optimizer2)

        scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                    gamma = self.params['scheduler_gamma'])
        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                    gamma = self.params['scheduler_gamma_2'])
        scheds.append(scheduler)
        scheds.append(scheduler2)
        return optims, scheds