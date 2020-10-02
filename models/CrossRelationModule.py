# -*- coding: utf-8 -*-
# @Time : 2020/10/2 4:57 下午
# @Author : ligang
# @FileName: CrossRelationModule.py
# @Email   : ucasligang@163.com
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as F

from models.AEAModule import AEAModule


class CrossRelationModule(nn.Module):
    def __init__(self, inplanes, transfer_name='W', scale_value=30, atten_scale_value=50, from_value=0.5,
                 value_interval=0.3):
        super(CrossRelationModule, self).__init__()

        self.inplanes = inplanes
        self.scale_value = scale_value

        if transfer_name == 'W':
            self.W = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            raise RuntimeError

        self.attention_layer = AEAModule(self.inplanes, atten_scale_value, from_value, value_interval)

    def forward(self, query_data, support_data):
        b, c, h, w = query_data.size()
        s, _, _, _ = support_data.size()
        support_data = support_data.unsqueeze(0).expand(b, -1, -1, -1, -1).contiguous().view(b * s, c, h, w)
        w_query = self.W(query_data).view(b, c, h * w)
        w_query = w_query.permute(0, 2, 1).contiguous()
        w_support = self.W(support_data).view(b, s, c, h * w).permute(0, 2, 1, 3).contiguous().view(b, c, s * h * w)
        w_query = F.normalize(w_query, dim=2)
        w_support = F.normalize(w_support, dim=1)

        f_x = torch.matmul(w_query, w_support)
        attention_score = self.attention_layer(w_query, f_x)

        query_data = query_data.view(b, c, h * w).permute(0, 2, 1)
        support_data = support_data.view(b, s, c, h * w).permute(0, 2, 1, 3).contiguous().view(b, c, s * h * w)
        query_data = F.normalize(query_data, dim=2)
        support_data = F.normalize(support_data, dim=1)

        match_score = torch.matmul(query_data, support_data)  # relation map
        attention_match_score = torch.mul(attention_score, match_score).view(b, h * w, s, h * w).permute(0, 2, 1, 3)

        final_local_score = torch.sum(attention_match_score.contiguous().view(b, s, h * w, h * w), dim=-1)
        final_score = torch.mean(final_local_score, dim=-1) * self.scale_value

        return final_score, final_local_score