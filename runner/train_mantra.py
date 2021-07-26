import os
import random
import sys
sys.path.append('..')

from torch import nn
import numpy as np
import pandas as pd
import torch
from l5kit.configs import load_config_data
from torch import optim
from tqdm import tqdm
from datetime import datetime

from model.autoencoder import MemoryAutoEncoder
from model.controller import MemoryController
from model.mantra import Mantra
from model.resnet18_with_memory import Resnet18WithMemory
from utils.dataset_loader import load_dataset, pytorch_neg_multi_log_likelihood_batch

import dataset_pytorch
from torch.utils.data import DataLoader
np.seterr(divide='ignore', invalid='ignore')


def forward( model, past, future, scene_one_hot, device, criterion=pytorch_neg_multi_log_likelihood_batch):
    bs, num_pred, _ = future.size()
    scene_one_hot = scene_one_hot.permute(0, 3, 1, 2).contiguous().to(device)
    preds, confidences = model(past, scene_one_hot, future)
    target_avails = torch.ones((bs, num_pred), device=device)
    # print(preds.size())
    loss = criterion(future, preds, confidences, target_avails)
    return loss, preds, confidences


def parse_data(data, device):
    history_avails = data['history_availabilities']
    history_avails = history_avails[:, :, None]
    history_positions = data['history_positions'] * history_avails
    history_coordinate = torch.cat((data['history_yaws'], history_positions), 2).to(device)

    target_avails = data['target_availabilities']
    target_avails = target_avails[:, :, None]
    target_positions = data['target_positions'] * target_avails
    targets = torch.cat((data['target_yaws'], target_positions), 2).to(device)

    input_image = data['image'].to(device)

    return input_image, history_coordinate, targets


def train_safe(past, future, map_input, model, device, criterion, optimizer):
    try :
        return train(past, future, map_input, model, device, criterion, optimizer)
    except RuntimeError as e:
        print(e)


def train(past, future, map_input, model, device, criterion, optimizer):
    # with torch.autograd.set_detect_anomaly(True):
    model.train()

    loss, preds, confidences = forward( model, past, future, map_input, device, criterion )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def eval(test_loader, model, device, criterion):
    model.eval()
    ades = []
    # progress_bar = tqdm(test_loader)
    for step, (index, past, future, scene_one_hot, video, clazz, num_vehicles, step, scene) in enumerate(test_loader):
        future = future.to(device)
        past = past.to(device)
        ade = eval_helper(past, future, scene_one_hot, model, device, criterion)
        ades.append(ade)
    ades = np.concatenate(ades, axis = 0)
    ades = np.mean(ades, axis= 0)

    return None, ades


def get_preds_max_confidence( preds, confidence):
    bs, num_modes = confidence.size()
    max_idx = torch.argmax(confidence, dim=1)
    max_preds = []

    for i in range(bs):
        max_pred = preds[i, max_idx[i], :, :]
        max_preds.append(max_pred)
    return torch.stack(max_preds)


def MSE(y_pred, y_gt, criterion):
    bs, num_pred, _ = y_pred.size()
    # y_pred  = y_pred.cpu().detach().numpy()
    # y_gt  = y_gt.cpu().detach().numpy()
    result = []
    for i in range(num_pred):
        mse = criterion(y_pred[:,i], y_gt[:,i])
        mse = mse.cpu().detach().numpy()
        result.append(mse)
    return np.array(result)

def eval_helper_(past, future, map_input, model, device, criterion):
    # with torch.autograd.set_detect_anomaly(True):
    model.eval()
    bs, num_pred, _ = future.size()
    map_input = map_input.permute(0, 3, 1, 2).contiguous().to(device)
    preds, confidences = model(past, future, map_input)
    preds = get_preds_max_confidence( preds, confidences )
    mse_result = MSE( preds, future, criterion )
    mse_result = np.sqrt(mse_result)
    ade = mse_result

    return ade

def eval_helper(past, future, map_input, model, device, criterion):
    # with torch.autograd.set_detect_anomaly(True):
    model.eval()
    bs, _, _ = future.size()
    map_input = map_input.permute(0, 3, 1, 2).contiguous().to(device)
    pred, confidences = model(past, map_input)
    num_pred = pred.size(1)
    future_rep = future.unsqueeze(1).repeat(1, num_pred, 1, 1)
    distances = torch.norm(pred - future_rep , dim=3)
    mean_distances = torch.mean(distances, dim=2)
    index_min = torch.argmin(mean_distances, dim=1)
    distance_pred = distances[torch.arange(0, len(index_min)), index_min]

    return distance_pred.cpu().detach().numpy()


def save(cfg, model, curr_iter, tr_loss, val_loss, ade):
    output_folder = cfg['output']['final_decoder_output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # model.save(folder_iter, curr_iter)
    torch.save(model.state_dict(), f'{output_folder}/{curr_iter}.pth')
    model.memory_auto_encoder.memory.save(cfg['memory_path'])
    csv_file = f"{output_folder}/../result_mantra.csv"
    if curr_iter == 0:
        # ade_header = ["ade_"+x for x in range(40)]
        # header =
        results = pd.DataFrame(data=[], columns=['iterations', 'tr_loss', 'val_loss', 'memory_size', 'ade_1', 'ade_2', 'ade_3', 'ade_4','fde_1','fde_2','fde_3','fde_4'])
        results.to_csv(csv_file, index=False)
    else:
        memory_size = model.memory_auto_encoder.memory.size()
        ade_1 = np.mean(ade[:9])
        ade_2 = np.mean(ade[:19])
        ade_3 = np.mean(ade[:29])
        ade_4 = np.mean(ade[:39])
        fde_1 = ade[9]
        fde_2 = ade[19]
        fde_3 = ade[29]
        fde_4 = ade[39]

        data = [curr_iter, tr_loss, val_loss, memory_size, ade_1, ade_2, ade_3, ade_4, fde_1, fde_2, fde_3, fde_4]
        results = pd.DataFrame(data=[data])
        results.to_csv(csv_file, index=False, mode='a', header=False)


def get_last_iter(cfg):
    output_folder = cfg['output']['final_decoder_output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_file = f"{output_folder}/../result_mantra.csv"
    if not os.path.exists(csv_file):
        results = pd.DataFrame(data=[],
                               columns=['iterations', 'tr_loss', 'val_loss', 'memory_size', 'ade_1', 'ade_2', 'ade_3',
                                        'ade_4', 'fde_1', 'fde_2', 'fde_3', 'fde_4'])
        results.to_csv(csv_file, index=False)
        return 0, None
    else:

        result_df = pd.read_csv(csv_file)
        curr_iter = int(result_df.iterations.iloc[-1])

        weight_path =  f'{output_folder}/{curr_iter}.pth'
        return curr_iter, weight_path


def main():
    print("mantra")
    cfg_file = "model.yaml"
    if (len(sys.argv) > 1):
        cfg_file = sys.argv[1]
    cfg = load_config_data(cfg_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    criterion = pytorch_neg_multi_log_likelihood_batch

    # load mantra
    model = Mantra(cfg, device = device)
    model.to(device)

    # check last iter
    curr_iter, weight_path = get_last_iter(cfg)
    print(weight_path)
    if weight_path:
        model.load_state_dict(torch.load(weight_path))

    if cfg["memory_path"]:
        if os.path.exists(cfg["memory_path"]):
            model.memory_auto_encoder.memory.load(cfg["memory_path"])


    losses_train = []
    batch_size = cfg['train_data_loader']['batch_size']
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])

    data_train = dataset_pytorch.TrackDataset('dataset_kitti_train.json')
    train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=1, shuffle=True)
    data_test = dataset_pytorch.TrackDataset('dataset_kitti_test.json')
    test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=1, shuffle=False)
    epoch = cfg["train_params"]["max_num_steps"]
    ii = 0
    criterion_eval = nn.MSELoss()

    for i in range(epoch):
        now = datetime.now()
        print("now =", now)
        progress_bar = tqdm(train_loader)
        for step, (index, past, future, scene_one_hot, video, clazz, num_vehicles, step, scene) in enumerate(
                progress_bar):
            future = future.to(device)
            past = past.to(device)
            loss = train(past, future, scene_one_hot, model, device, criterion, optimizer)
            # loss = train_safe(past, future, scene_one_hot, model, device, criterion, optimizer)
            if loss == None :
                continue
            losses_train.append(loss.item())
            # losses_val.append(val_loss.item())
            progress_bar.set_description(f"train_loss: {loss.item()}")
            ii += 1
            if ii % cfg['train_params']['checkpoint_every_n_steps'] == 0:

                print(f'\n {curr_iter + ii}')
                val_loss, ade = eval(test_loader, model, device, criterion_eval)
                save(cfg, model, curr_iter + ii, np.mean(losses_train[-100:]), val_loss, ade)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    main()
