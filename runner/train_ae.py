from torch.utils.data import DataLoader
import dataset_pytorch
import os
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.autoencoder import AutoEncoder
from utils import pytorch_neg_multi_log_likelihood_single, load_config_data

def forward(model, past, future, _, device, criterion=pytorch_neg_multi_log_likelihood_single):
    bs, num_pred, _ = future.size()
    past_encoded = model.past_encoder(past)
    future_encoded = model.future_encoder(future)

    last_input = past[:, -1, :].flatten(1)
    preds = model.future_decoder(last_input, past_encoded, future_encoded)
    preds = preds.permute(1, 0, 2).contiguous()
    target_avails = torch.ones((bs,num_pred), device= device)
    loss = criterion(future, preds, target_avails)
    return loss, preds


def train_forward(model, past, future, scene_one_hot, device, criterion):
    model.past_encoder_optimizer.zero_grad()
    model.future_encoder_optimizer.zero_grad()
    model.decoder_optimizer.zero_grad()

    loss, preds = forward(model, past, future, scene_one_hot, device, criterion)
    loss.backward()

    model.past_encoder_optimizer.step()
    model.future_encoder_optimizer.step()
    model.decoder_optimizer.step()

    return loss, preds


def parse_data(data, device):
    history_avails = data['history_availabilities']
    history_avails = history_avails[:, :, None]
    history_positions = data['history_positions'] * history_avails
    inputs = torch.cat((data['history_yaws'], history_positions), 2).to(device)

    target_avails = data['target_availabilities']
    target_avails = target_avails[:, :, None]
    target_positions = data['target_positions'] * target_avails
    targets = torch.cat((data['target_yaws'], target_positions), 2).to(device)

    target_availabilities = data["target_availabilities"].to(device)

    return inputs, targets, target_availabilities


def train(past, future, scene_one_hot,  model, device, criterion):
    model.train()
    torch.set_grad_enabled(True)
    loss, _ = train_forward(model, past, future, scene_one_hot, device, criterion)
    return loss


def eval(test_loader, model, device, criterion):
    model.eval()
    losses = []
    for step, (index, past, future, scene_one_hot, video, clazz, num_vehicles, step, scene) in enumerate(test_loader):
        future = future.to(device)
        past = past.to(device)
        loss, preds = forward(model, past, future, scene_one_hot, device, criterion)
        losses.append(loss.item())
    return np.mean( losses )


def save(model, curr_iter, tr_loss, val_loss):
    ae_output_folder = model.cfg['output']['ae_output_folder']
    folder_iter = f'{ae_output_folder}/{curr_iter}'
    if not os.path.exists(folder_iter):
        os.makedirs(folder_iter)

    torch.save(model.past_encoder.state_dict(), f'{folder_iter}/past_encoder.pth')
    torch.save(model.future_encoder.state_dict(), f'{folder_iter}/future_encoder.pth')
    torch.save(model.future_decoder.state_dict(), f'{folder_iter}/future_decoder.pth')
    csv_file = f"{folder_iter}/../../result_autoencoder.csv"
    if curr_iter == 0:
        results = pd.DataFrame(data=[], columns=['iterations', 'tr_loss', 'val_loss'])
        results.to_csv(csv_file, index=False)
    else:
        results = pd.DataFrame(data=[[curr_iter, tr_loss, val_loss]])
        results.to_csv(csv_file, index=False, mode='a', header=False)


def main():
    print("train_autoencoder_kitti")
    cfg_file = "model.yaml"
    if (len(sys.argv) > 1):
        cfg_file = sys.argv[1]
    cfg = load_config_data(cfg_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    batch_size = cfg['train_data_loader']['batch_size']

    data_train = dataset_pytorch.TrackDataset('dataset_kitti_train.json')
    train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=1, shuffle=True)
    data_test = dataset_pytorch.TrackDataset('dataset_kitti_test.json')
    test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=1, shuffle=False)



    criterion = pytorch_neg_multi_log_likelihood_single
    model = AutoEncoder(cfg, input_dim=2, device= device)
    model.load_weight()

    losses_train = []
    progress_bar = tqdm(train_loader)
    # losses_val = []
    curr_iter = cfg["input"]["ae_last_iter"]
    epoch = cfg["train_params"]["max_num_steps"]
    ii = 0
    for i in range( epoch ):
        for step, (index, past, future, scene_one_hot, video, clazz , num_vehicles, step, scene) in enumerate(progress_bar):
            future = future.to(device)
            past = past.to(device)
            loss = train(past, future, scene_one_hot, model, device, criterion)
            losses_train.append(loss.item())
            # losses_val.append(val_loss.item())
            progress_bar.set_description(f"train_loss: {loss.item()}")
            ii += 1
            if ii % cfg['train_params']['checkpoint_every_n_steps'] == 0:
                val_loss = eval(test_loader, model, device, criterion)
                save( model, curr_iter + ii, np.mean( losses_train[-100:] ), val_loss )


if __name__ == '__main__':
    main()