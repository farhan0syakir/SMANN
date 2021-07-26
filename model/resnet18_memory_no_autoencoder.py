import os
from typing import Dict

import torch
from torch import nn
from torchvision.models.resnet import resnet18



class Memory(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.keys = torch.rand(3, 11 * 3, device=self.device)
        self.values = torch.rand(3, 50, 3, device=self.device)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def read(self, queries, top_num=3):
        # print(queries.size()) #bs, 10, 3
        queries = queries.flatten(1)
        bs, n = queries.size()
        results = []
        mem_size = self.keys.size(0)
        for query in queries:
            query = query.unsqueeze(0)
            query = query.expand(mem_size, n)
            result = self.cos(self.keys, query)
            _, index = result.topk(top_num)  # index size 3
            result = self.values[index]
            results.append(result)
        # print(result.size()) # 50, 3
        results = torch.stack(results)
        # bs, 3, 50, 48
        # print(results.size()) #bs, 50, 3
        return results

    def write(self, key, value):
        # self.memory = torch.Tensor(1, 1, 1)
        self.keys = torch.cat((self.keys, key.flatten().unsqueeze(0)), 0)
        self.values = torch.cat((self.values, value.unsqueeze(0)), 0)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.keys, path+'/keys.pth')
        torch.save(self.values, path+'/values.pth')

    def load(self, path):
        self.keys = torch.load(path+'/keys.pth')
        self.values = torch.load(path + '/values.pth')



class Model(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        # architecture = cfg["model_params"]["model_architecture"]
        # architecture = 'resnet18'
        backbone = resnet18(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets

        backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features + 50 * 3 * 3, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

        self.memory = Memory()

    def load_memory(self, path):
        pass

    def save_memory(self, path):
        pass

    def forward(self, x, memory_pred):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        memory_pred = torch.flatten(memory_pred, 1)
        x = torch.cat((x, memory_pred), 1)

        x = self.head(x)
        x = self.logit(x)

        # todo cat history
        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences