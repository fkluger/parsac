import torch
from torch import nn
import torch.nn.functional as F

class CNNet(nn.Module):

    def __init__(self, input_dim, output_dim, blocks=5, batch_norm=True, separate_weights=True):

        super(CNNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.p_in = nn.Conv1d(self.input_dim, 128, 1, 1, 0)

        self.res_blocks = []

        self.batch_norm = batch_norm
        self.separate_probs = separate_weights

        for i in range(0, blocks):
            if batch_norm:
                self.res_blocks.append((
                    nn.Conv1d(128, 128, 1, 1, 0),
                    nn.BatchNorm1d(128),
                    nn.Conv1d(128, 128, 1, 1, 0),
                    nn.BatchNorm1d(128),
                ))
            else:
                self.res_blocks.append((
                    nn.Conv1d(128, 128, 1, 1, 0),
                    nn.Conv1d(128, 128, 1, 1, 0),
                ))

        for i, r in enumerate(self.res_blocks):
            super(CNNet, self).add_module(str(i) + 's0', r[0])
            super(CNNet, self).add_module(str(i) + 's1', r[1])
            if batch_norm:
                super(CNNet, self).add_module(str(i) + 's2', r[2])
                super(CNNet, self).add_module(str(i) + 's3', r[3])

        self.p_out = nn.Conv1d(128, output_dim, 1, 1, 0)
        if self.separate_probs:
            self.p_out2 = nn.Conv1d(128, output_dim, 1, 1, 0)


    def forward(self, inputs):
        '''
        Forward pass.

        inputs -- 3D data tensor (BxNxC)
        '''
        inputs_ = torch.transpose(inputs, 1, 2)

        x = inputs_[:, 0:self.input_dim]

        x = F.relu(self.p_in(x))

        for r in self.res_blocks:
            res = x
            if self.batch_norm:
                x = F.relu(r[1](F.instance_norm(r[0](x))))
                x = F.relu(r[3](F.instance_norm(r[2](x))))
            else:
                x = F.relu(F.instance_norm(r[0](x)))
                x = F.relu(F.instance_norm(r[1](x)))
            x = x + res

        log_ng = F.logsigmoid(self.p_out(x))
        log_ng = torch.transpose(log_ng, 1, 2)
        normalizer = torch.logsumexp(log_ng, dim=-1, keepdim=True)
        log_ng = log_ng - normalizer

        if self.separate_probs:
            log_ng2 = F.logsigmoid(self.p_out2(x))
            log_ng2 = torch.transpose(log_ng2, 1, 2)
            normalizer = torch.logsumexp(log_ng2, dim=-2, keepdim=True)
            log_ng2 = log_ng2 - normalizer
        else:
            log_ng2 = log_ng

        return log_ng, log_ng2
