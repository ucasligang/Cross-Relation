import sys

import torch
from torch import nn

from models.CrossRelationModule import CrossRelationModule
from models.backbone import CNNEncoder
from utils.utils import init_weights

sys.dont_write_bytecode = True

class CrossRelationNet(nn.Module):
    def __init__(self, base_model='Conv64F', base_model_info=None, **kwargs):
        super(CrossRelationNet, self).__init__()

        if base_model_info is None:
            base_model_info = {}
        self.base_model = base_model
        self.base_model_info = base_model_info
        self.kwargs = kwargs
        self._init_module()

    def _init_module(self):
        if self.base_model == 'Conv64F':
            self.features = CNNEncoder(**self.base_model_info)
        else:
            raise RuntimeError

        self.metric_layer = CrossRelationModule(**self.kwargs)

        init_weights(self, init_type='normal')

    def forward(self, query_data, support_data):
        query_feature = self.features(query_data)
        support_feature = []
        for support in support_data:
            support_feature.append(self.features(support))

        scores = []
        local_scores = []
        for support in support_feature:
            score, local_score = self.metric_layer(query_feature, support)
            scores.append(score)
            local_scores.append(local_score)

        scores = torch.cat(scores, 1)
        local_scores = torch.cat(local_scores, 1)

        return query_feature, scores, local_scores


if __name__ == '__main__':
    model = CrossRelationNet(base_model='Conv64F', inplanes=64)
    query_data = torch.rand(75, 3, 84, 84)
    support_data = [torch.rand(25, 3, 84, 84)]
    result = model(query_data, support_data)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  + Number of params: %.3fM' % (trainable_num / 1e6))
    print('  + Number of params: {}'.format(trainable_num))
