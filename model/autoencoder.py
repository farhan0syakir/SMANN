import torch
from torch import nn, optim

from .memory import MantraMemory


class PastEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=48, device='cpu'):
        super(PastEncoder, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._gru = nn.GRU(input_dim, hidden_dim, 1)
        self._device = device

    def forward(self, input_history):
        """
        param:
            input_history: bs, history 2
        output:
            hidden: 2, bs, hidden_dim
        """
        batch_size = input_history.size(0)
        input_history = input_history.permute(1, 0, 2).contiguous()
        hidden = torch.zeros(1, batch_size, self._hidden_dim, device=self._device)
        _, hidden = self._gru(input_history, hidden)
        return hidden


class FutureEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=48, device='cpu'):
        super(FutureEncoder, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._gru = nn.GRU(input_dim, hidden_dim, 1)
        self._device = device

    def forward(self, future_input):
        """
        param:
            future_input: future_dim, bs, 2
        output:
            rep: 2, bs, hidden_dim
        """
        future_input = future_input.permute(1, 0, 2).contiguous()
        batch_size = future_input.size(1)
        hidden = torch.zeros(1, batch_size, self._hidden_dim, device=self._device)
        _, hidden = self._gru(future_input, hidden)
        return hidden


class FutureDecoder(nn.Module):
    def __init__(self, seq_len, output_dim=3, embedding_dim=32, hidden_dim=96, num_layers=1, dropout=0.0):  # 61, 96, 2
        super(FutureDecoder, self).__init__()
        self._seq_len = seq_len
        self._hidden_dim = hidden_dim
        self._embedding_dim = embedding_dim

        self._decoder = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self._spatial_embedding = nn.Linear(output_dim, embedding_dim)
        self._hidden2pos = nn.Linear(hidden_dim, output_dim)

    def forward(self, last_pos_rel, past_encoded,
                future_encoded):  # last_pos_rel, state_tuple, start_pos=None, start_vel=None
        """
        Inputs:
        - last_pos_rel: Tensor of shape (batch, 3)
        - past_encoded: Tensor of shape(1, batch, 48)
        - future_encoded: Tensor of shape(1, batch, 48)
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos_rel.size(0)
        pred_traj = []
        decoder_input = self._spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self._embedding_dim)
        state = torch.cat((past_encoded, future_encoded), 2)
        for _ in range(self._seq_len):
            output, state = self._decoder(decoder_input, state)
            rel_pos = self._hidden2pos(output.view(-1, self._hidden_dim))
            embedding_input = rel_pos
            decoder_input = self._spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self._embedding_dim)
            pred_traj.append(rel_pos.view(batch, -1))

        pred_traj = torch.stack(pred_traj, dim=0)
        return pred_traj


class AutoEncoder(nn.Module):
    def __init__(self, cfg, input_dim = 3, device="cpu"):
        super(AutoEncoder, self).__init__()
        self.device = device
        self.cfg = cfg
        self.hidden_size = cfg['model_params']['ae_hidden_size']
        self.history_num_frames = cfg['model_params']['history_num_frames']
        self.future_num_frames = cfg['model_params']['future_num_frames']
        self.model_name = cfg['model_params']['model_name']
        self.past_encoder = PastEncoder(hidden_dim=self.hidden_size, input_dim=input_dim, device=device)
        self.future_encoder = FutureEncoder(hidden_dim=self.hidden_size, input_dim=input_dim, device=device)
        self.future_decoder = FutureDecoder(self.future_num_frames, output_dim=input_dim, hidden_dim=2 * self.hidden_size)

        self.past_encoder_optimizer = optim.Adam(self.past_encoder.parameters(), lr=0.0001)
        self.future_encoder_optimizer = optim.Adam(self.future_encoder.parameters(), lr=0.0001)
        self.decoder_optimizer = optim.Adam(self.future_decoder.parameters(), lr=0.0001)


    def load_weight(self):
        if (self.cfg['input']['ae_past_encoder_path']):
            self.past_encoder.load_state_dict(torch.load(self.cfg['input']['ae_past_encoder_path'], map_location=self.device))
            self.future_encoder.load_state_dict(torch.load(self.cfg['input']['ae_future_encoder_path'],map_location=self.device))
            self.future_decoder.load_state_dict(torch.load(self.cfg['input']['ae_future_decoder_path'],map_location=self.device))

    def predict(self, inputs):
        pass


class MemoryAutoEncoder(nn.Module):
    def __init__(self, cfg, input_dim = 3, device="cpu"):
        super(MemoryAutoEncoder, self).__init__()
        self.hidden_size = cfg['model_params']['ae_hidden_size']
        self.history_num_frames = cfg['model_params']['history_num_frames']
        self.future_num_frames = cfg['model_params']['future_num_frames']
        self.device = device
        self.memory = MantraMemory(device=device).to(device)
        self.auto_encoder = AutoEncoder(cfg, input_dim=input_dim, device=device)
        self.topk = 1

    def load_auto_encoder(self):
        # todo
        self.auto_encoder.load_weight()

    def load_memory(self, path):
        self.memory.load(path)

    def eval(self):
        self.auto_encoder.eval()

    def forward(self, inputs):
        # todo
        batch_size = inputs.size(1)
        last_history = inputs[:, -1, :].squeeze(0)

        past_encoded = self.auto_encoder.past_encoder(inputs)
        past_encoded = past_encoded.squeeze(0)

        future_encodeds = self.memory.read(past_encoded, self.topk)
        past_encoded = past_encoded.unsqueeze(0)

        preds = []
        for i in range(self.topk):
            future_encoded = future_encodeds[:, i, :].flatten(1).unsqueeze(0)
            pred = self.auto_encoder.future_decoder(last_history, past_encoded, future_encoded)
            preds.append(pred)

        preds = torch.stack(preds).squeeze(0)
        return preds, past_encoded
