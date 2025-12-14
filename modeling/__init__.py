import torch
from .config import _C as cfg

from .diffusion_model import DiffusionMSE
from .basicGEBD import BasicGEBD

import modeling.time_transformer as transformer

def build_encoder(cfg):
    if cfg.MODEL.ENCODER == 'basicGEBD':
        return BasicGEBD(cfg)
    else:
        raise NotImplemented(f'No such encoder: {cfg.MODEL.ENCODER}')

def build_decoder(cfg):
    if cfg.MODEL.DECODER == 'transformer':
        return transformer.DiffDecoder(cfg)
    else:
        raise NotImplemented(f'No such decoder: {cfg.MODEL.DECODER}')

def build_model(cfg):
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)

    if cfg.MODEL.NAME == 'DiffusionMSE':
        model = DiffusionMSE(cfg, encoder, decoder)
    elif cfg.MODEL.NAME == 'DiffusionMSE_CFG':
        model = DiffusionMSE(cfg, encoder, decoder, guidance=True)

    else:
        raise NotImplemented(f'No such model: {cfg.MODEL.NAME}')

    if cfg.MODEL.SYNC_BN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
