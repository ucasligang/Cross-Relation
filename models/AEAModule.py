# -*- coding: utf-8 -*-
# @Time : 2020/10/2 5:00 下午
# @Author : ligang
# @FileName: AEAModule.py
# @Email   : ucasligang@163.com
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as F


class AEAModule(nn.Module):
    def __init__(self, inplanes, scale_value=50, from_value=0.4, value_interval=0.5):
        super(AEAModule, self).__init__()
        self.inplanes = inplanes
        self.scale_value = scale_value
        self.from_value = from_value
        self.value_interval = value_interval

        self.f_psi = nn.Sequential(
            nn.Linear(self.inplanes, self.inplanes // 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.inplanes // 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, f_x):
        # the Eq.(7) should be an approximation of Step Function with the adaptive threshold,
        # please refer to https://github.com/LegenDong/ATL-Net/pdf/ATL-Net_Update.pdf
        b, hw, c = x.size()
        clamp_value = self.f_psi(x.view(b * hw, c)) * self.value_interval + self.from_value
        clamp_value = clamp_value.view(b, hw, 1)
        clamp_fx = torch.sigmoid(self.scale_value * (f_x - clamp_value))
        attention_mask = F.normalize(clamp_fx, p=1, dim=-1)

        return attention_mask

