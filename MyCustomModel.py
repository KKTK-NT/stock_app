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
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]})
        intent_vector = torch.unsqueeze(action_embedding, 1)
        action_logits = torch.sum(avail_actions * intent_vector, axis=1)
        inf_mask = torch.clamp(torch.log(action_mask), min=torch.finfo(torch.float32).min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
