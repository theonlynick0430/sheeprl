import torch.nn as nn
from typing import Dict, Any
import hydra
from sheeprl.algos.dreamer_v3.agent import MLP
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights
import copy


class RND(nn.Module):
    def __init__(
        self,
        rnd_cfg: Dict[str, Any],
        latent_state_size: int,
    ):
        super().__init__()
        self.rnd_cfg = rnd_cfg
        self.latent_state_size = latent_state_size

        # create intrinsic critic
        critic_cfg = rnd_cfg.critic
        critic_ln_cls = hydra.utils.get_class(critic_cfg.layer_norm.cls)
        self.critic = MLP(
            input_dims=latent_state_size,
            output_dim=1,
            hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
            activation=hydra.utils.get_class(critic_cfg.dense_act),
            layer_args={"bias": critic_ln_cls == nn.Identity},
            flatten_dim=None,
            norm_layer=critic_ln_cls,
            norm_args={
                **critic_cfg.layer_norm.kw,
                "normalized_shape": critic_cfg.dense_units,
            },
        )

        # create target, predictor networks
        target_ln_cls = hydra.utils.get_class(rnd_cfg.layer_norm.cls)
        self.predictor = MLP(
            input_dims=latent_state_size,
            output_dim=rnd_cfg.output_dim,
            hidden_sizes=[rnd_cfg.dense_units] * rnd_cfg.mlp_layers,
            activation=hydra.utils.get_class(rnd_cfg.dense_act),
            layer_args={"bias": target_ln_cls == nn.Identity},
            flatten_dim=None,
            norm_layer=target_ln_cls,
            norm_args={
                **rnd_cfg.layer_norm.kw,
                "normalized_shape": rnd_cfg.dense_units,
            },
        )
        self.target = copy.deepcopy(self.predictor)

        # freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        # weight initialization
        self.critic.apply(init_weights)
        self.predictor.apply(uniform_init_weights(2.0))
        self.target.apply(uniform_init_weights(1.0))

    def get_target(self, x):
        return self.target(x)
    
    def get_prediction(self, x):
        return self.predictor(x)
    
    def get_intrinsic_reward(self, x):
        return ((self.get_target(x) - self.get_prediction(x)) ** 2).mean(axis=-1, keepdim=True)
    
    def get_value(self, x):
        return self.critic(x)