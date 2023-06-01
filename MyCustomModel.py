import torch
import numpy as np
from torch import nn
from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_torch

_, nn = try_import_torch()

class MaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5, 13, ), dtype=np.float32)
        # print("obs_space",self.obs_space.shape)
        
        # self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        true_obs_shape = int(obs_space.shape[0] - num_outputs)

        self.action_embed_model = FullyConnectedNetwork(spaces.Box(-np.inf, np.inf, shape=(true_obs_shape,), dtype=np.float32),
                                                        self.action_space, num_outputs, self.model_config, self.name + "_action_embedding")

    def forward(self, input_dict, state, seq_lens):
        print("action_mask",input_dict)
        action_mask = input_dict["obs"]["action_mask"]  # action_mask torch.Size([32, 5, 13])
        keys = input_dict['obs'].keys()
        flat_obs = torch.cat([input_dict['obs'][i].flatten(start_dim=1) for i in keys if i != 'action_mask'], dim=-1)
        action_embedding, _ = self.action_embed_model({'obs': flat_obs})
        bsize, stocks, action_dim = action_mask.shape

        # Reshape action_embedding to match the shape of action_mask
        action_embedding = action_embedding.view(bsize, stocks, action_dim)


        # Apply action_mask to filter out invalid actions
        action_logits = action_mask * action_embedding

        # Apply a large negative value to invalid actions
        action_logits = action_logits + (1 - action_mask) * -1e7
        # print("action_logits", action_logits.shape)
        
        action_logits = action_logits.view(-1, stocks * action_dim)
        
        return action_logits, state

    def value_function(self):
        return self.action_embed_model.value_function()
