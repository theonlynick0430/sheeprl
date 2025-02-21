from typing import Any, Dict, Tuple
import copy
from math import prod

import torch.nn as nn
import torch.nn.init as init
from lightning import Fabric
import hydra
import gymnasium

from sheeprl.models.models import MultiEncoder, MLP
from sheeprl.algos.ppo.agent import CNNEncoder, MLPEncoder


INTR = "intrinsic"
EXTR = "extrinsic"

def build_networks(
    fabric: Fabric,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
) -> Tuple[nn.Module, nn.Module]:
    cnn_keys=cfg.algo.cnn_keys.encoder
    mlp_keys=cfg.algo.mlp_keys.encoder
    encoder_cfg = cfg.algo.encoder
    rnd_cfg = cfg.algo.rnd
    in_channels = sum([prod(obs_space[k].shape[:-2]) for k in cnn_keys])
    mlp_input_dim = sum([obs_space[k].shape[0] for k in mlp_keys])
    screen_size=cfg.env.screen_size
    
    cnn_encoder = (
        CNNEncoder(in_channels, encoder_cfg.cnn_features_dim, screen_size, cnn_keys)
        if cnn_keys is not None and len(cnn_keys) > 0
        else None
    )
    mlp_encoder = (
        MLPEncoder(
            mlp_input_dim,
            encoder_cfg.mlp_features_dim,
            mlp_keys,
            encoder_cfg.dense_units,
            encoder_cfg.mlp_layers,
            hydra.utils.get_class(encoder_cfg.dense_act),
            encoder_cfg.layer_norm,
        )
        if mlp_keys is not None and len(mlp_keys) > 0
        else None
    )
    feature_extractor = MultiEncoder(cnn_encoder, mlp_encoder)
    head = MLP(
        input_dims=feature_extractor.output_dim,
        output_dim=rnd_cfg.k, # output embedding dim
    )

    target = nn.Sequential(feature_extractor, head)
    for param in target.parameters():
        # random init
        init.xavier_normal_(param)
        # freeze the target network
        param.requires_grad = False

    predictor = copy.deepcopy(target)
    for param in predictor.parameters():
        # random init
        if param.dim() > 1:
            init.xavier_normal_(param)
        else:
            init.zeros_(param) # studies show better to zero init bias
        # unfreeze the predictor network
        param.requires_grad = True

    target = fabric.setup_module(target)
    predictor = fabric.setup_module(predictor)

    return target, predictor