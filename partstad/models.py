import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CN_layer(nn.Module):
    def __init__(
        self,
        in_channel=256,
        out_channel=1,
        num_block=3,
        he_init=False,
    ):
        super().__init__()

        self.layer_in = nn.Linear(in_channel, in_channel)
        self.relu = nn.ReLU()
    
    def forward(self,feature):
        feature_in = self.layer_in(feature)
        feature_mean = torch.mean(feature_in, dim=0, keepdim=True) # [1, in_channel]
        feature_std = torch.std(feature_in, dim=0, keepdim=True, correction=0) #[1, in_channel]
        cn_feature = (feature_in - feature_mean)/feature_std
        cn_feature = self.relu(cn_feature)

        return cn_feature


class WeightPredNetworkCNe(nn.Module):
    def __init__(
        self,
        in_channel=256,
        out_channel=1,
        num_cn_layer=1,
        he_init=False,
        skip_connection=True,
    ):
        super().__init__()
        self.skip_connection = skip_connection

        self.CN_layers = nn.ModuleList([CN_layer(in_channel) for i in range(num_cn_layer)])
        
        self.layer_out = nn.Linear(in_channel,out_channel)

        if he_init:
            self.apply(self._init_weights_he)
        else:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.0001)
            module.bias.data.zero_()

    def _init_weights_he(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight.data)
            
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, feature):
        """
            feature : [N,in_channel]
        """

        feature_in = feature
        
        for layer in self.CN_layers:
            cn_feature = layer(feature_in)
            if self.skip_connection:
                feature_in = feature_in + cn_feature

        out = self.layer_out(feature_in)

        return out 
