import torch
import numpy as np
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_torch

_, nn = try_import_torch()

class MaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.action_embed_model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name + "_action_embedding")

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"] # avail_actions torch.Size([32, 13])
        action_mask = input_dict["obs"]["action_mask"] # action_mask torch.Size([32, 5, 13])
        bsize = avail_actions.size()[0]

        keys = input_dict['obs'].keys()
        flat_obs = torch.cat([input_dict['obs'][i].view(input_dict['obs'][i].size(0), -1) for i in keys], dim=-1)
        flat_obs = {'obs': flat_obs} # flat_obs torch.Size([32, 184])
    
        action_embedding, _ = self.action_embed_model(flat_obs)
        # action_embedding = torch.tensor(action_embedding) # action_embedding torch.Size([32, 65])
        intent_vector = action_embedding.unsqueeze(1).view(bsize, (len(keys) - 3), -1) # intent_vector torch.Size([32, 5, 13])

        # # Reshape to [32, 1, 13]
        # avail_actions_reshaped = avail_actions.unsqueeze(1)
        # # Repeat along dimension 1 to get [32, 5, 13]
        # avail_actions_repeated = avail_actions_reshaped.repeat(1, (len(keys) - 3), 1)

        # Reshape to [32, 1, 65]
        # expanded_avail_actions = avail_actions_repeated.view(bsize, 1, -1)

        action_logits = action_mask * intent_vector
        # print("action_logits",np.shape(action_logits)) # action_logits torch.Size([32, 5])
        # action_logits = action_logits.view(bsize, (len(keys) - 3), avail_actions_reshaped.size()[-1])
        # print("action_logits",np.shape(action_logits))
        
        # inf_mask = torch.log(action_mask.clamp(min=1e-9))
        # print("inf_mask",np.shape(inf_mask))
        return action_logits.view(bsize, -1), state






    def value_function(self):
        return self.action_embed_model.value_function()
