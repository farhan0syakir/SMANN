import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(self, history_len, future_len, map_hidden_dim, num_modes, input_dim=2, device='cpu'):
        super(TransformerDecoder, self).__init__()
        self.history_len = history_len
        self.future_len = future_len
        # self.num_modes = num_modes
        # encoder_dim = 48
        # decoder_dim = map_hidden_dim + encoder_dim + input_dim * future_len
        encoder_layers = TransformerEncoderLayer(d_model=66, nhead=6)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 6)
        # self.hidden2pos = nn.Linear(decoder_dim, num_modes * input_dim)
        # self.pos2embed = nn.Linear(num_modes * input_dim, encoder_dim)
        self.num_preds =  2 * self.future_len * num_modes
        self.logit = nn.Linear(60 * 66, out_features=self.num_preds + num_modes)
        self.device = device
        self.src_mask = None
        self.num_modes = num_modes

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, past, future, map_input, has_mask=True):
        mask_size = self.history_len + self.future_len
        if has_mask:
            device = past.device
            if self.src_mask is None or self.src_mask.size(0) != mask_size:
                mask = self._generate_square_subsequent_mask(mask_size).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #padding here
        map_input = map_input.squeeze().unsqueeze(1)
        map_input = map_input.expand(-1, self.history_len + self.future_len, -1)
        past_future = torch.cat((past, future),1)
        output = torch.cat((past_future, map_input), 2)
        output = output.permute(1, 0, 2).contiguous()
        output = self.transformer_encoder(output, self.src_mask)
        output = output.permute(1, 0, 2).contiguous()
        output = torch.flatten(output, 1)
        output = self.logit(output)
        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = output.shape
        pred, confidences = torch.split(output, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


class MantraLSTMDecoder(nn.Module):
    def __init__(self, history_len, future_len, map_hidden_dim, num_modes, input_dim = 2, device = 'cpu'):
        self.history_len = history_len
        super(MantraLSTMDecoder, self).__init__()
        self.future_len = future_len
        self.num_modes = num_modes
        encoder_dim = 48
        decoder_dim = map_hidden_dim + encoder_dim + input_dim * future_len
        self.lstmEncoder = nn.LSTM(2, 48)
        self.lstmDecoder = nn.LSTM(encoder_dim, decoder_dim)
        self.hidden2pos = nn.Linear(decoder_dim, num_modes * input_dim)
        self.pos2embed = nn.Linear(num_modes * input_dim, encoder_dim)

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_modes * future_len * input_dim, out_features=num_modes),
            nn.ReLU(inplace=False)
        )
        self.device = device

    def combine_hidden(self, h0, c0, future, map_input):
        h0 = h0.permute(1,0,2).contiguous()
        c0 = c0.permute(1,0,2).contiguous()
        h0 = torch.flatten(h0, 1)
        c0 = torch.flatten(c0, 1)

        future = torch.flatten(future, 1)
        map_input = map_input.squeeze()
        h0 = torch.cat((h0, future, map_input), 1)
        c0 = torch.cat((c0, future, map_input), 1)

        h0 = h0.unsqueeze(0)
        c0 = c0.unsqueeze(0)
        return h0, c0

    def forward(self, past, future, map_input):
        bs = past.size(0)
        past = past.permute(1,0,2).contiguous()
        # init state
        h0 = torch.zeros((1, bs, 48), device = self.device)
        c0 = torch.zeros((1, bs, 48), device = self.device)

        output, (h0, c0) = self.lstmEncoder(past, (h0, c0))
        decoder_input = output[-1].unsqueeze(0)
        predictions = []
        h0, c0 = self.combine_hidden(h0, c0, future, map_input)
        for i in range(self.future_len):
            decoder_input, (h0, c0) = self.lstmDecoder(decoder_input, (h0, c0))
            prediction = self.hidden2pos(decoder_input.squeeze())
            decoder_input = self.pos2embed(prediction)
            decoder_input = decoder_input.unsqueeze(0)
            predictions.append(prediction)

        predictions = torch.stack(predictions)
        predictions = predictions.permute(1,2, 0).contiguous()
        predictions = predictions.view(bs, self.num_modes, self.future_len, 2)
        confidences = self.fc(torch.flatten(predictions,1))
        confidences = torch.softmax(confidences, dim=1)
        return predictions, confidences


class MantraDecoder(nn.Module):
    def __init__(self, history_len, future_len, map_hidden_dim, num_modes, input_dim = 2):
        super(MantraDecoder, self).__init__()
        self.future_len = future_len
        self.history_len = history_len
        self.input_len = future_len * input_dim + history_len * input_dim + map_hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_len, out_features=512),
            nn.ReLU(inplace=False)
        )
        num_modes = num_modes
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        self.logit = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, out_features=self.num_preds + num_modes),
            nn.ReLU(inplace=False)
        )

    def forward(self, past, future, map):
        past = torch.flatten(past, 1)
        future = torch.flatten(future, 1)
        map = map.squeeze()
        x = torch.cat((past, future, map), 1)
        x = self.fc(x)
        x = self.logit(x)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

class StackedTransformerDecoder(nn.Module):
    def __init__(self, history_len, future_len, map_hidden_dim, num_modes, input_dim=2, device='cpu',   hidden=66, n_layers=12, attn_heads=6, dropout=0.1 ):
        super(StackedTransformerDecoder, self).__init__()
        self.history_len = history_len
        self.future_len = future_len
        # self.num_modes = num_modes
        # encoder_dim = 48
        # decoder_dim = map_hidden_dim + encoder_dim + input_dim * future_len
        # multi-layers transformer blocks, deep network
        self.encs = nn.ModuleList(
            [TransformerEncoderLayer(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        # self.enc1 = TransformerEncoderLayer(d_model=66, nhead=6)
        # self.enc2 = TransformerEncoderLayer(d_model=66, nhead=6)

        self.decs = nn.ModuleList(
            [TransformerDecoderLayer(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])


        # self.transformer_encoder = TransformerEncoder(encoder_layers, 6)
        # self.hidden2pos = nn.Linear(decoder_dim, num_modes * input_dim)
        # self.pos2embed = nn.Linear(num_modes * input_dim, encoder_dim)
        self.num_preds =  2 * self.future_len * num_modes
        self.logit = nn.Linear(60 * 66, out_features=self.num_preds + num_modes)
        self.device = device
        self.src_mask = None
        self.num_modes = num_modes

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, past, future, map_input, has_mask=True):
        mask_size = self.history_len + self.future_len
        if has_mask:
            device = past.device
            if self.src_mask is None or self.src_mask.size(0) != mask_size:
                mask = self._generate_square_subsequent_mask(mask_size).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None


        #padding here
        map_input = map_input.squeeze().unsqueeze(1)
        map_input = map_input.expand(-1, self.history_len + self.future_len, -1)
        past_future = torch.cat((past, future),1)
        output = torch.cat((past_future, map_input), 2)
        output = output.permute(1, 0, 2).contiguous()

        for transformer in self.encs:
            output = transformer.forward(output)

        mem = output
        for transformer in self.decs:
            output = transformer.forward(output, mem)

        # output = self.transformer_encoder(output, self.src_mask)
        output = output.permute(1, 0, 2).contiguous()
        output = torch.flatten(output, 1)
        output = self.logit(output)
        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = output.shape
        pred, confidences = torch.split(output, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences