import torch
import random
import time
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from gat_utils import accuracy, predicts_vec
from gat_net import Gat_Net


def train_epoch(model, optimizer, device, batch_loaders, batch_matrices, batch_targets):
    model.train()
    epoch_train_loss, epoch_train_acc, nb_data, i = 0, 0, 0, 0
    for i in range(len(batch_loaders)):
        batch_graph = batch_loaders[i].to(device)
        batch_matrix = batch_matrices[i].to(device)
        batch_target = batch_targets[i].to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graph, batch_matrix)
        loss = model.loss(batch_scores, batch_target)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_target)
        nb_data += batch_target.size(0)
    epoch_train_loss /= (i + 1)
    epoch_train_acc /= nb_data

    return epoch_train_loss, epoch_train_acc, optimizer


def test_epoch(model, device, batch_loaders, batch_matrices, batch_targets):
    model.eval()
    epoch_test_loss, epoch_test_acc, nb_data, i = 0, 0, 0, 0
    with torch.no_grad():
        for i in range(len(batch_loaders)):
            batch_graph = batch_loaders[i].to(device)
            batch_matrix = batch_matrices[i].to(device)
            batch_target = batch_targets[i].to(device)

            batch_scores = model.forward(batch_graph, batch_matrix)
            loss = model.loss(batch_scores, batch_target)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_target)
            nb_data += batch_target.size(0)

            test_vec = predicts_vec(batch_scores.detach())

        epoch_test_loss /= (i + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc, test_vec


def train_pipeline(gat_params, gat_net_params, data, exp_test_data):
    t0 = time.time()
    per_epoch_time = []

    device = gat_net_params['device']

    # setting seeds
    random.seed(gat_params['seed'])
    np.random.seed(gat_params['seed'])
    torch.manual_seed(gat_params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(gat_params['seed'])

    model = Gat_Net(gat_net_params)
    model = model.to(device)

    model_parameters = {}
    for name, parameters in model.named_parameters():
        model_parameters[name] = parameters.size()

    optimizer = optim.Adam(model.parameters(), lr=gat_params['init_lr'],
                           weight_decay=gat_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=gat_params['lr_reduce_factor'],
                                                     patience=gat_params['lr_schedule_patience'], verbose=True)

    # At any point you can hit Ctrl + C to break out of training early.
    print("Gat training......")
    try:
        with tqdm(range(gat_params['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, data[0][0],
                                                                           data[0][1], data[0][2])
                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'], train_loss=epoch_train_loss)
                per_epoch_time.append(time.time() - start)
                scheduler.step(epoch_train_loss)
                if optimizer.param_groups[0]['lr'] < gat_params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                # Stop training after gat_msa_params['max_time'] hours
                if time.time() - t0 > gat_params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(gat_params['max_time']))
                    break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    # Clean the dirty nodes
    clean_nodes_vec = {}
    for i in range(len(exp_test_data)):
        data_node_id = exp_test_data[i][3]
        _, _, vec = test_epoch(model, device, exp_test_data[i][0], exp_test_data[i][1], exp_test_data[i][2])
        clean_nodes_vec.update({data_node_id: vec})

    return clean_nodes_vec
