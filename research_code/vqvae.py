import os
import math
from argparse import ArgumentParser, Namespace

import numpy as np

from torchvision.utils import make_grid

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from scipy.cluster.vq import kmeans2

import datasets

# Credit goes to Andrej Kaparthy, from whose code this is adapted
# -----------------------------------------------------------------------------
class NormalLoss:
    """
    simple normal distribution with fixed variance, as used by DeepMind in their VQVAE
    note that DeepMind's reconstruction loss (I think incorrectly?) misses a factor of 2,
    which I have added to the normalizer of the reconstruction loss in nll(), we'll report
    number that is half of what we expect in their jupyter notebook
    """
    data_variance = 0.06327039811675479 # cifar-10 data variance, from deepmind sonnet code

    @classmethod
    def inmap(cls, x):
        return x - 0.5 # map [0,1] range to [-0.5, 0.5]

    @classmethod
    def unmap(cls, x):
        return torch.clamp(x + 0.5, 0, 1)

    @classmethod
    def nll(cls, x, mu):
        return ((x - mu)**2).mean() / (2 * cls.data_variance)

class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out

class Encoder(nn.Module):

    def __init__(self, input_channels=3, n_hid=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
        )

        self.output_channels = 2 * n_hid
        self.output_stide = 4

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self, n_init=32, n_hid=64, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class Quantizer(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.kld_scale = 10.0

        self.proj = nn.Conv2d(num_hiddens, embedding_dim, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)
        #print(self.embed.weight.shape) # n_embed x embedding_dim

        self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, z, proj=True):
        B, C, H, W = z.size()

        # project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
        if proj:
            z_e = self.proj(z)
        else:
            z_e = z
        z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
        flatten = z_e.reshape(-1, self.embedding_dim)

        # DeepMind def does not do this but I find I have to... ;\
        if self.training and self.data_initialized.item() == 0:
            print('running kmeans!!') # data driven initialization for the embeddings
            rp = torch.randperm(flatten.size(0))
            kd = kmeans2(flatten[rp[:20000]].data.cpu().numpy(), self.n_embed, minit='points')
            self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        dist = self.get_dist(flatten)
        _, ind = (-dist).max(1)
        ind = einops.rearrange(ind, '(B H W) -> B H W', B=B, H=H, W=W)
        log_priors = nn.functional.log_softmax(einops.rearrange((-dist), '(B H W) D -> B D H W', B=B, H=H, W=W), dim=1)

        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind) # (B, H, W, C)
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        z_q = z_q.permute(0, 3, 1, 2) # stack encodings into channels again: (B, C, H, W)
        return z_q, diff, ind, log_priors

    def get_dist(self, flat_z):
        '''
        returns distance from z to each embedding vec
        flat_z should be of shape (B*H*W, C), e.g. (10*16*16, 256)
        '''
        dist = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        return dist

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)
    
    def embed_one_hot(self, embed_vec):
        '''
        embed vec is of shape (B * T * H * W, n_embed)
        '''
        return embed_vec @ self.embed.weight

class VQVAE(pl.LightningModule):

    def __init__(self, n_hid, embedding_dim, num_embeddings, input_channels=3, log_perplexity=False, perplexity_freq=500):
        super().__init__()
        self.save_hyperparameters()

        # encoder/decoder module pair
        self.encoder = Encoder(input_channels=input_channels, n_hid=n_hid)
        self.decoder = Decoder(n_init=embedding_dim, n_hid=n_hid, output_channels=input_channels)

        # the quantizer module sandwiched between them, +contributes a KL(posterior || prior) loss to ELBO
        self.quantizer = Quantizer(self.encoder.output_channels, num_embeddings, embedding_dim)

        # the data reconstruction loss in the ELBO
        self.recon_loss = NormalLoss

    def forward(self, x):
        z = self.encoder(x)
        z_q, latent_loss, ind = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, ind
    
    def encode_with_grad(self, x):
        '''
        Use this method if you want to finetune the encoder in downstream tasks, otherwise use self.encode_only
        '''
        z = self.encoder(self.recon_loss.inmap(x))
        z_q, diff, ind, log_priors = self.quantizer(z)
        return z_q, diff, ind, log_priors
    
    def training_step(self, batch, batch_idx):
        print(f'{batch.shape = }')
        # center image
        img = self.recon_loss.inmap(batch)
        
        # forward pass
        img_hat, latent_loss, ind = self.forward(img)
        
        # compute reconstruction loss
        recon_loss = self.recon_loss.nll(img, img_hat)
        
        # loss = reconstruction_loss + codebook loss from quantizer
        loss = recon_loss + latent_loss
        
        # logging
        self.log('loss', loss, on_step=True)
        if self.hparams.log_perplexity:
            if (self.global_step + 1) % self.hparams.perplexity_freq == 0:
                self.eval()
                perplexity, cluster_use = self._compute_perplexity(ind)
                self.train()
                self.log('perplexity', perplexity, prog_bar=True)
                self.log('cluster_use', cluster_use, prog_bar=True)
        return loss

    @torch.no_grad()
    def _compute_perplexity(self, ind):
        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(ind, self.quantizer.n_embed).float().reshape(-1, self.quantizer.n_embed)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        return perplexity, cluster_use
                
    def configure_optimizers(self):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 1e-4},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        self.optimizer = optimizer

        return optimizer

    # ---------- inference-only methods -----------
    @torch.no_grad()
    def reconstruct_only(self, x):
        z = self.encoder(self.recon_loss.inmap(x))
        z_q, *_ = self.quantizer(z)
        x_hat = self.decoder(z_q)
        x_hat = self.recon_loss.unmap(x_hat)
        return x_hat
    
    @torch.no_grad()
    def decode_only(self, z_q):
        x_hat = self.decoder(z_q)
        x_hat = self.recon_loss.unmap(x_hat)
        return x_hat

    @torch.no_grad()
    def encode_only(self, x):
        z = self.encoder(self.recon_loss.inmap(x))
        z_q, _, ind, log_priors = self.quantizer(z)
        return z_q, ind, log_priors
    # ----------------------------------------------
    
# -----------------------------------------------------------------------------
def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

"""
These ramps/decays follow DALL-E Appendix A.2 Training https://arxiv.org/abs/2102.12092
"""
class DecayTemperature(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
        t = cos_anneal(0, 150000, 1.0, 1.0/16, trainer.global_step)
        pl_module.quantizer.temperature = t

class RampBeta(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        t = cos_anneal(0, 5000, 0.0, 5e-4, trainer.global_step)
        pl_module.quantizer.kld_scale = t

class DecayLR(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, 1200000, 3e-4, 1.25e-6, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t

class GenerateCallback(pl.Callback):
    def __init__(self, batch_size=6, dataset=None, save_to_disk=False, every_n_batches=100, precision=32):
        """
        Inputs:
            batch_size - Number of images to generate
            dataset - Dataset to sample from
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_batches = every_n_batches
        self.save_to_disk = save_to_disk
        self.initial_loading = False
        self.img_batch = dataset.iter.buffered_batch_iter(batch_size)

    def on_batch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (pl_module.global_step+1) % self.every_n_batches == 0:
            self.reconstruct(trainer, pl_module, pl_module.global_step+1)

    def reconstruct(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        if self.img_batch.device != pl_module.device:
            self.img_batch = self.img_batch.to(pl_module.device)

        reconstructed_img = pl_module.reconstruct_only(self.img_batch)

        images = torch.stack([self.img_batch, reconstructed_img], dim=1).reshape((self.batch_size * 2, *self.img_batch.shape[1:]))

        # log images to tensorboard
        trainer.logger.experiment.add_image('Reconstruction',make_grid(images, nrow=2), epoch)


def main():
    pl.seed_everything(1337)

    # -------------------------------------------------------------------------
    # arguments...
    parser = ArgumentParser()
    # training related
    parser.add_argument('--log_freq', type=int, default=10, help='How often to save values to the logger')
    parser.add_argument('--save_freq', type=int, default=500, help='Save the model every N training steps')
    parser.add_argument('--progbar_rate', type=int, default=1, help='How often to update the progress bar in the command line interface')
    parser.add_argument('--callback_batch_size', type=int, default=6, help='How many images to reconstruct for callback (shown in tensorboard/images)')
    parser.add_argument('--callback_freq', type=int, default=100, help='How often to reconstruct for callback (shown in tensorboard/images)')
    # model size
    parser.add_argument("--num_embeddings", type=int, default=512, help="vocabulary size; number of possible discrete states")
    parser.add_argument("--embedding_dim", type=int, default=32, help="size of the vector of the embedding of each discrete token")
    parser.add_argument("--n_hid", type=int, default=64, help="number of channels controlling the size of the model")
    # dataloader related
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--env_name", type=str, default='MineRLNavigate-v0')
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=6)
    # model loading args
    parser.add_argument('--load_from_checkpoint', default=False, action='store_true')
    parser.add_argument('--version', default=None, type=int, help='Version of model, if training is resumed from checkpoint')
    #other args
    parser.add_argument('--log_dir', type=str, default='./run_logs')
    # done!
    args = parser.parse_args()
    # -------------------------------------------------------------------------

    # make sure that relevant dirs exist
    run_name = f'PretrainedVQVAE/{args.env_name}'
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    # init model
    vqvae_kwargs = {
        'n_hid': args.n_hid,
        'num_embeddings': args.num_embeddings,
        'embedding_dim': args.embedding_dim
    }
    if args.load_from_checkpoint:
        checkpoint_file = os.path.join(log_dir, 'lightning_logs', f'version_{args.version}', 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint_file}')
        model = VQVAE.load_from_checkpoint(checkpoint_file, **vqvae_kwargs)
    else:
        model = VQVAE(**vqvae_kwargs)

    # load data
    train_data = datasets.VQVAEDataset(args.env_name, args.data_dir, args.batch_size)
    train_loader = DataLoader(train_data, num_workers=args.num_workers)

    # create callbacks
    callbacks = []
    
    # model-saving callback
    callbacks.append(ModelCheckpoint(monitor='loss', mode='min', save_last=True, every_n_train_steps=args.save_freq))
    
    # callback to sample reconstructed images
    callbacks.append(
        GenerateCallback(
            batch_size=args.callback_batch_size, 
            dataset=train_data, 
            save_to_disk=False, 
            every_n_batches=args.callback_freq
        )
    )
    
    # annealing schedules for lots of constants
    callbacks.append(DecayLR())
    
    # create trainer instance
    trainer = pl.Trainer(
        callbacks=callbacks, 
        default_root_dir=log_dir, 
        gpus=torch.cuda.device_count(),
        max_epocs=args.epochs,
        accelerator='dp',
        log_every_n_steps=args.log_freq,
        progress_bar_refresh_rate=args.progbar_rate
    )
    
    # train model
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
