import os
import random
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm

from model.autoencoder import MemoryAutoEncoder
from model.controller import MemoryController
import dataset_pytorch
from torch.utils.data import DataLoader
from utils import load_config_data


def loss_controller(controller_output, xf, xf_hat, device='cpu'):
    controller_output = torch.flatten(controller_output)
    e = get_reconstruction_error(xf, xf_hat, device)
    loss = e * (1 - controller_output) + (1 - e) * controller_output
    return torch.mean(loss)


def get_reconstruction_error(xf, xf_hat, device='cpu'):
    N = xf.size(1)
    thresholds = torch.tensor(np.linspace(0.0, 4.0, num=N), device=device)  # 0.5 meter per pixel
    distances = torch.sqrt(torch.sum((xf - xf_hat) ** 2, 2))
    ii = (distances) > thresholds
    e = 1 - torch.sum(ii, 1) / N
    return e


def parse_data(data, device):
    history_avails = data['history_availabilities']
    history_avails = history_avails[:, :, None]
    history_positions = data['history_positions'] * history_avails
    inputs = torch.cat((data['history_yaws'], history_positions), 2).to(device)

    target_avails = data['target_availabilities']
    target_avails = target_avails[:, :, None]
    target_positions = data['target_positions'] * target_avails
    targets = torch.cat((data['target_yaws'], target_positions), 2).to(device)

    return inputs, targets


def train(past, future, memoryAutoEncoder, model, device, criterion, optimizer):

    xf = future
    with torch.no_grad():
        memoryAutoEncoder.eval()
        xf_hat, past_encoded = memoryAutoEncoder.predict(past)
        xf_hat = xf_hat.permute(1, 0, 2)

    model.train()

    output = model(xf, xf_hat)
    # write to memory
    batch_size = output.size(0)
    is_write = output > random.random()
    past_encoded = past_encoded.squeeze(0)
    gt_future_encoded = memoryAutoEncoder.auto_encoder.future_encoder(future)
    # print(gt_future_encoded.size())
    # exit()
    for i in range(batch_size):
        if is_write[i]:
            memoryAutoEncoder.memory.write(past_encoded[i], gt_future_encoded[:, i, :].flatten())

    is_reset_memory = random.randint(0, 99) < 2  # 2% change reset memory
    if (is_reset_memory):
        memoryAutoEncoder.memory.reset()

    loss = criterion(output, xf, xf_hat, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def eval(test_loader, memoryAutoEncoder, model, device, criterion):
    model.eval()
    losses = []
    for step, (index, past, future, scene_one_hot, video, clazz, num_vehicles, step, scene) in enumerate(test_loader):
        future = future.to(device)
        past = past.to(device)
        loss = eval_helper(past, future, memoryAutoEncoder, model, device, criterion)
        losses.append(loss.item())
    return np.mean(losses)


def eval_helper(past, future, memory_auto_encoder, model, device, criterion):
    with torch.no_grad():
        model.eval()
        xf = future
        xf_hat, past_encoded = memory_auto_encoder.predict(past)
        xf_hat = xf_hat.permute(1, 0, 2)
        output = model(xf, xf_hat)

        loss = criterion(output, xf, xf_hat, device)
        return loss


def save(memory_auto_encoder, model, curr_iter, tr_loss, val_loss, cfg):
    memory_output_path = cfg['memory_path']
    folder_iter = cfg['output']['memory_controller_output_folder']
    if not os.path.exists(folder_iter):
        os.makedirs(folder_iter)

    torch.save(model.state_dict(), f'{folder_iter}/{curr_iter}.pth')
    memory_auto_encoder.memory.save(memory_output_path)

    csv_file = f"{folder_iter}/../result_memory_controller.csv"
    if curr_iter == 0:
        results = pd.DataFrame(data=[], columns=['iterations', 'tr_loss', 'val_loss', 'memory_size'])
        results.to_csv(csv_file, index=False)
    else:
        memory_size = memory_auto_encoder.memory.size()
        results = pd.DataFrame(data=[[curr_iter, tr_loss, val_loss, memory_size]])
        results.to_csv(csv_file, index=False, mode='a', header=False)


def main():
    print("train_memory_controller")
    cfg_file = "model.yaml"
    if (len(sys.argv) > 1):
        cfg_file = sys.argv[1]
    cfg = load_config_data(cfg_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    criterion = loss_controller
    # auto_encoder_path = 'output/AutoEncoderTmp/model/0'
    memory_auto_encoder = MemoryAutoEncoder(cfg, input_dim=2, device= device)
    memory_auto_encoder.auto_encoder.load_weight()
    # memory_auto_encoder.load_auto_encoder(auto_encoder_path)
    # memory_auto_encoder.eval()

    model = MemoryController(cfg, input_dim = 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])

    curr_iter = 0
    weight_path = cfg["input"]["memory_controller_path"]
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
        curr_iter = cfg["input"]["memory_controller_last_iter"]

    losses_train = []
    batch_size = cfg['train_data_loader']['batch_size']

    data_train = dataset_pytorch.TrackDataset('dataset_kitti_train.json')
    train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=1, shuffle=True)
    data_test = dataset_pytorch.TrackDataset('dataset_kitti_test.json')
    test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=1, shuffle=False)
    progress_bar = tqdm(train_loader)
    epoch = cfg["train_params"]["max_num_steps"]
    ii = 0
    for i in range(epoch):
        for step, (index, past, future, scene_one_hot, video, clazz, num_vehicles, step, scene) in enumerate(progress_bar):
            future = future.to(device)
            past = past.to(device)
            loss = train(past, future, memory_auto_encoder, model, device, criterion, optimizer)
            losses_train.append(loss.item())
            # losses_val.append(val_loss.item())
            progress_bar.set_description(f"train_loss: {loss.item()}")
            ii += 1
            if ii % cfg['train_params']['checkpoint_every_n_steps'] == 0:
                val_loss = eval(test_loader, memory_auto_encoder, model, device, criterion)
                save(memory_auto_encoder, model, curr_iter + ii, np.mean(losses_train[-100:]), val_loss, cfg)

    # todo
    # test_loss = ??


if __name__ == "__main__":
    main()
