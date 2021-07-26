import os
import random

import torch
from torch import nn

from .autoencoder import MemoryAutoEncoder
from .controller import MemoryController
from .decoder import MantraLSTMDecoder, TransformerDecoder, MantraDecoder, StackedTransformerDecoder
from .memory import MantraMemory
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MapEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MapEncoder, self).__init__()
        # 8 X (k3; s2; p1); 16 X (k3; s1; p1)
        k3 = 3
        s2 = 2
        p1 = 1
        s1 = 1
        self.linear1 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(input_size, hidden_size, k3, stride=s2, padding=p1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )
        ])
        for i in range(8):
            tmp = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, k3, stride=s2, padding=p1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
            )
            self.linear1.append(tmp)

        self.linear2 = nn.ModuleList([])
        for i in range(16):
            tmp = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, k3, stride=s1, padding=p1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
            )
            self.linear2.append(tmp)

    def forward(self, x):
        for l in self.linear1:
            x = l(x)

        for l in self.linear2:
            x = l(x)

        return x


class Mantra(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super(Mantra, self).__init__()
        self.device = device
        self.input_feature_size = 2
        self.model_name = cfg['model_params']['model_name']
        self.hidden_size = cfg['model_params']['ae_hidden_size']
        self.future_num_frames = cfg['model_params']['future_num_frames']
        self.history_num_frames = cfg['model_params']['history_num_frames']
        self.num_modes = cfg['model_params']['num_modes']
        self.topk = cfg['model_params']['topk']
        self.image_dim = 4
        self.memory_auto_encoder = MemoryAutoEncoder(cfg, input_dim=2, device=device)
        self.memory_controller = MemoryController(cfg, input_dim=2)
        self.map_encoder = MapEncoder(self.image_dim, 64).to(device)
        if(cfg['model_params']['final_decoder'] == 'MantraLSTMDecoder'):
            self.decoder = MantraLSTMDecoder(self.history_num_frames, self.future_num_frames, 64, self.num_modes, device=device).to(device)
        elif (cfg['model_params']['final_decoder'] == 'TransformerDecoder'):
            self.decoder = TransformerDecoder(self.history_num_frames, self.future_num_frames, 64, self.num_modes).to(device)
        elif (cfg['model_params']['final_decoder'] == 'StackedTransformerDecoder'):
            self.decoder = StackedTransformerDecoder(self.history_num_frames, self.future_num_frames, 64, self.num_modes).to(
                device)
        else:
            self.decoder = MantraDecoder(self.history_num_frames, self.future_num_frames, 64, self.num_modes).to(device)

        self.load(cfg)
        self.memory_auto_encoder.eval()

    def train(self):
        self.map_encoder.train()
        self.decoder.train()

    def eval(self):
        self.map_encoder.eval()
        self.decoder.eval()

    def save(self, folder_iter, curr_iter):
        torch.save(self.map_encoder.state_dict(), f'{folder_iter}/{curr_iter}_map.pth')
        torch.save(self.decoder.state_dict(), f'{folder_iter}/{curr_iter}_dec.pth')
        torch.save(self.memory.state_dict(), self.memory_path)

    def load(self, cfg):
        if cfg['input']['ae_past_encoder_path']:
            self.memory_auto_encoder.auto_encoder.load_weight()

        if cfg['input']['memory_controller_path']:
            self.memory_controller.load_state_dict(torch.load(cfg['input']['memory_controller_path'],map_location=self.device))



    def forward(self, past, data_map, future = None):
        """
        data_histrory: bs, history, 2
        data_map: bs, 3, 244, 244
        targets: bs, future, 2

        """
        past = past.detach()
        preds, past_encoded = self.memory_auto_encoder(past)
        past_encoded = past_encoded.detach()
        preds = preds.detach()
        # change from predict to forward
        preds = preds.permute(1, 0, 2).contiguous()
        # memory_contrller
        if future is not None:
            output = self.memory_controller(future, preds)
            # write to memory
            batch_size = output.size(0)
            is_write = output > random.random()
            past_encoded = past_encoded.squeeze(0)
            gt_future_encoded = self.memory_auto_encoder.auto_encoder.future_encoder(future)

            for i in range(batch_size):
                if is_write[i]:
                    self.memory_auto_encoder.memory.write(past_encoded[i], gt_future_encoded[:, i, :].flatten())

        map_encoded = self.map_encoder(data_map)
        final_preds, confidences = self.decoder(past, preds, map_encoded )

        return final_preds, confidences
