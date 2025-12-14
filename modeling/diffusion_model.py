import itertools
import math

import numpy as np
import einops
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d
from utils.distribute import is_main_process

########## General Functions ##########
def normalize_scale(x, scale):
    return ((x / scale) + 1.) / 2.

def denormalize_scale(x, scale):
    return (x * 2. - 1.) * scale

def prepare_gaussian_targets(targets, sigma=1):
    targets = targets.squeeze(1)
    gaussian_targets = []
    for batch_idx in range(targets.shape[0]):
        t = targets[batch_idx]
        axis = torch.arange(len(t), device=targets.device)
        gaussian_t = torch.zeros_like(t)
        indices, = torch.nonzero(t, as_tuple=True)
        for i in indices:
            g = torch.exp(-(axis - i) ** 2 / (2 * sigma * sigma))
            gaussian_t += g

        gaussian_t = gaussian_t.clamp(0, 1)
        # gaussian_t /= gaussian_t.max()
        gaussian_targets.append(gaussian_t)
    gaussian_targets = torch.stack(gaussian_targets, dim=0)
    return gaussian_targets.unsqueeze(1)

########## Diffusion Functions ##########
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def beta_linear_log_snr(t):
    return -torch.log(torch.expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionMSE(Module):
    def __init__(self, cfg, encoder, decoder, guidance=False):
        super().__init__()
        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        self.fpn_start_idx = cfg.MODEL.FPN_START_IDX
        self.head_choice = cfg.MODEL.HEAD_CHOICE
        self.dim = cfg.MODEL.DIMENSION
        self.guidance = guidance
        self.num_layers = self.head_choice - self.fpn_start_idx + 1

        if self.backbone_name == 'resnet50':
            self.backbone = getattr(models, self.backbone_name)(pretrained=True, norm_layer=FrozenBatchNorm2d)
            in_feat_dim = self.backbone.fc.in_features
            for param in itertools.chain(self.backbone.conv1.parameters(), self.backbone.bn1.parameters()):
                param.requires_grad = False

            del self.backbone.fc
            self.fpn_layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4][:self.head_choice+1]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.x1_out = nn.Identity() if self.dim == in_feat_dim // 8 else nn.Sequential(nn.Conv1d(in_feat_dim // 8, self.dim, 1), nn.ReLU(inplace=True)) # 256
        self.x2_out = nn.Identity() if self.dim == in_feat_dim // 4 else nn.Sequential(nn.Conv1d(in_feat_dim // 4, self.dim, 1), nn.ReLU(inplace=True)) # 512
        self.x3_out = nn.Sequential(nn.Conv1d(in_feat_dim // 2, self.dim, 1), nn.ReLU(inplace=True)) # 1024 -> dim
        self.x4_out = nn.Sequential(nn.Conv1d(in_feat_dim, self.dim, 1), nn.ReLU(inplace=True)) # 2048 -> dim
        self.x_out = [self.x1_out, self.x2_out, self.x3_out, self.x4_out]

        
            
        self.freeze = cfg.MODEL.BACKBONE.FREEZE
        self.gaus_target = cfg.INPUT.GAUSSIAN_TARGET
        self.encoder = encoder
        self.decoder = decoder
        
        self.enc_out = cfg.MODEL.ENCODER_OUT
        if self.enc_out:
            self.encoder_out = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv1d(self.dim, 1, kernel_size=1))
            
        self.device = 'cuda'
        
        self.gaus_sigma = cfg.INPUT.GAUS_SIGMA
        self.only_gaus_target = cfg.INPUT.ONLY_TARGET_GAUS
        #################################### Diffusion parameters ####################################
        self.cfg_prob = cfg.DIFFUSION.CFG_PROB
        self.cfg_scale = cfg.DIFFUSION.CFG_SCALE
        timesteps = cfg.DIFFUSION.TIMESTEPS
        if cfg.DIFFUSION.BETA_SCHEDULE == 'cosine':
            betas = cosine_beta_schedule(timesteps)  # torch.Size([1000])
        elif cfg.DIFFUSION.BETA_SCHEDULE == 'linear':
            betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = cfg.DIFFUSION.SAMPLING_TIMESTEPS
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = cfg.DIFFUSION.DDIM_SAMPLING_ETA
        self.scale = cfg.DIFFUSION.SNR_SCALE
        
        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))
        
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
    

    def backbone_forward(self, imgs):
        if isinstance(imgs, torch.Tensor):
            B, T = imgs.shape[:2]
        else:
            B, T = imgs[0].shape[:2]

        if not self.freeze:
            imgs = einops.rearrange(imgs, 'b t c h w -> (b t) c h w')
            x = self.extract_features(imgs)
        
            if self.fpn_layers is not None:
                xs = []   # feature pyramid, max_len = len(self.fpn_layers)
                for i, layer in enumerate(self.fpn_layers):
                    x = layer(x)
                    if 'resnet' in self.backbone_name or 'tsm' in self.backbone_name:
                        xi = x
                    else:
                        xi = einops.rearrange(x, 'b c t h w -> (b t) c h w')

                    xi = self.avg_pool(xi).squeeze(-1).squeeze(-1)
                    xi = einops.rearrange(xi, "(b t) c -> b c t", b=B)
                    xi = self.x_out[i](xi)
                    xs.append(xi)

            xs.append(xi)
            x_fp = torch.cat(xs, dim=1) # resnet intermediate layer 512 dim concat
            x_fps = einops.rearrange(x_fp, "b (nl c) t -> b nl c t", b=B, c=self.dim, nl=len(xs)) # nl : number of layers
            x_fps = x_fps[:,self.fpn_start_idx: self.head_choice+1]
            return x_fps
        else:
            xs = []
            for i in range(self.fpn_start_idx, self.head_choice+1):
                x = imgs[i]
                x = einops.rearrange(x, 'b t c -> b c t')
                xi = self.x_out[i](x)
                xs.append(xi)

            x_fp = torch.cat(xs, dim=1) # resnet intermediate layer 512 dim concat
            x_fps = einops.rearrange(x_fp, "b (nl c) t -> b nl c t", b=B, c=self.dim, nl=len(xs)) # nl : number of layers
            return x_fps

    def extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        return x

    def forward(self, imgs, targets=None, sampling_timesteps=None):
        """
        Args:
            imgs: imgs (B, T, C, H, W);
            targets: (B, T);
        Returns:
        """
        backbone_feat = self.backbone_forward(imgs) # B nl C T
        targets = targets.unsqueeze(1).to(backbone_feat.dtype)

        B, nl, C, T = backbone_feat.shape
        zero_cond = False
        if self.guidance:
            if self.training and torch.rand(1) < self.cfg_prob:
                cond = torch.zeros(B, C, T).to(self.device)
                zero_cond = True
            else:
                cond = self.encoder(backbone_feat)
        else:
            cond = self.encoder(backbone_feat)
        
        if len(cond.shape) == 3:
            cond = cond.permute(2,0,1) # T B C

        if self.training:    
            if self.gaus_target:
                loss_targets = prepare_gaussian_targets(targets, sigma=self.gaus_sigma)
            else:
                loss_targets = targets
            

            loss_targets = denormalize_scale(loss_targets, self.scale)
            noised_targets, noises, ts = self.prepare_targets(loss_targets)
            
            logits = self.decoder(noised_targets, cond, ts) # B 1 T

            logits = logits / self.scale
            
            if self.only_gaus_target:
                assert(self.gaus_target==False)
                loss_targets = prepare_gaussian_targets(targets, sigma=self.gaus_sigma) # 0 ~ 1
                loss_targets = denormalize_scale(loss_targets, self.scale) # [-s, s]
            
            loss_targets = loss_targets / self.scale

            logits = logits.to(torch.float32)
            loss_targets = loss_targets.to(torch.float32)
            loss = F.mse_loss(logits, loss_targets)

            if not zero_cond and self.enc_out:
                cond = einops.rearrange(cond, 't b c -> b c t')
                enc_out = self.encoder_out(cond)
                enc_out = enc_out.to(torch.float32)
                loss_enc = F.mse_loss(enc_out, loss_targets)
                loss = loss + loss_enc

            loss_dict = {'loss': loss}
            return loss_dict
        else:
            scores = self.ddim_sample(cond, sampling_timesteps) # [B, T]
        return scores
    
    def prepare_targets(self, targets):
        batch_size = targets.size(0)

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)

        noise = torch.randn(size=targets.shape, device=self.device)

        x_start = targets  # [-scale, +scale]

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        return x, noise, t

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None): # forward diffusion
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def ddim_sample(self, cond, sampling_timesteps):
        if len(cond.shape) == 3:
            T, B, C = cond.shape
        else:
            B, nl, C, T = cond.shape    
        
        
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps if sampling_timesteps==None else sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        shape = (B, 1, T)
        x_time = torch.randn(shape, device=self.device)
        x_start = None
        
        for time, time_next in time_pairs:
            
            time_cond = torch.full((B,), time, device=self.device, dtype=torch.long)
            
            if self.guidance:
                pred_noise, x_start, pred_label = self.model_predictions_cfg(x_time, cond, time_cond)
            else:
                pred_noise, x_start, pred_label = self.model_predictions(x_time, cond, time_cond)
            
            x_return = torch.clone(x_start) # [-s, s]
            
            if time_next < 0:
                x_time = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_time)

            x_time = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        
        x_return = normalize_scale(x_return, self.scale) # [0, 1]
        x_return = torch.clamp(x_return, 0, 1)

        return x_return.flatten(1)
    
    def model_predictions(self, x, cond, t):
        logits = self.decoder(x, cond, t)
        pred_label = logits
        
        x_start = torch.clamp(pred_label, min=-self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        return pred_noise, x_start, pred_label

    def model_predictions_cfg(self, x, cond, t):
        cond_pred = self.decoder(x, cond, t)
        uncond_pred = self.decoder(x, torch.zeros_like(cond), t)

        final_pred = cond_pred + (cond_pred - uncond_pred) * self.cfg_scale

        final_pred = torch.clamp(final_pred, -self.scale, self.scale)  # [-scale, scale]
        uncond_pred = torch.clamp(uncond_pred, -self.scale, self.scale)
        cond_pred = torch.clamp(cond_pred, -self.scale, self.scale)

        x_start = final_pred
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start, cond_pred
