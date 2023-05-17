import torch
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
        # print(input_dict)
        # print(input_dict["obs"]["action_mask"])
        # print(input_dict["obs"])
        # print(input_dict["obs"]["state"])
        action_mask = input_dict["obs"]["action_mask"]
        action_logits, _ = self.action_embed_model({"obs": input_dict["obs"]})
        # action_logits = torch.sum(action_embedding, axis=1)
        inf_mask = torch.where(action_mask == 0, float('-inf'), 0)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
