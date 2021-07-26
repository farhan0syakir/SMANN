import torch
from torch import nn


class MantraMemory(nn.Module):
    keys = None
    values = None

    def __init__(self, hidden_size=48, device='cpu'):
        super(MantraMemory, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.reset()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def read(self, queries, top_num=3):
        # print(queries.size()) #bs, 96
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
        # print(result.size()) # 3, 96
        results = torch.stack(results)
        # bs, 3, 50, 48
        # print(results.size()) #bs, 3, 96
        return results

    def write(self, key, value):
        # self.memory = torch.Tensor(1, 1, 1)
        # if duplicate return
        if key in self.keys:
            return
        self.keys = torch.cat((self.keys, key.unsqueeze(0)), 0)
        self.values = torch.cat((self.values, value.unsqueeze(0)), 0)

    def save(self, path):
        tmp = torch.stack([self.keys, self.values])
        torch.save(tmp, path)

    def load(self, path):
        self.keys, self.values = torch.load(path,map_location=self.device)

    def test(self):
        print(self.keys.size())
        print(self.values.size())

    def reset(self):
        self.keys = torch.rand(10, self.hidden_size, device=self.device)
        self.values = torch.rand(10, self.hidden_size, device=self.device)

    def size(self):
        return self.keys.size(0)
