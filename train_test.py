import os
import json, torch

import numpy as np
import torch.backends.cudnn
import torch.utils.data


from model import MGEL
from metrics import model_evaluate

from utils import argparser, DTADataset, GraphDataset, collate, predicting, read_data, train, setup_seed
import warnings

warnings.filterwarnings('ignore')


def create_dataset_for_train_test(affinity, dataset, fold):
    # load dataset
    dataset_path = '../data/' + dataset + '/'

    train_fold_origin = json.load(open(dataset_path + 'train_set.txt'))
    train_folds = []
    for i in range(len(train_fold_origin)):
        if i != fold:
            train_folds += train_fold_origin[i]
    test_fold = json.load(open(dataset_path + 'test_set.txt')) if fold == -100 else train_fold_origin[fold]


    rows, cols = np.where(np.isnan(affinity) == False)
    train_rows, train_cols = rows[train_folds], cols[train_folds]

    train_Y = affinity[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

    test_rows, test_cols = rows[test_fold], cols[test_fold]
    test_Y = affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    return train_dataset, test_dataset


def train_test():
    FLAGS = argparser()

    dataset = FLAGS.dataset
    cuda_name = f'cuda:{FLAGS.cuda_id}'
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TEST_BATCH_SIZE = FLAGS.batch_size
    NUM_EPOCHS = FLAGS.num_epochs
    LR = FLAGS.lr

    Architecture = [MGEL][FLAGS.model]

    fold = FLAGS.fold


    model_name = Architecture.__name__
    if fold != -100:
        model_name += f"-{fold}"

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("Epochs:", NUM_EPOCHS)
    print('batch size', TRAIN_BATCH_SIZE)
    print("Learning rate:", LR)
    print("Fold", fold)
    print("Model name:", model_name)
    print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)

    print("\ncreate dataset ......")

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    affinity = read_data(dataset)

    train_data, test_data = create_dataset_for_train_test(affinity, dataset, fold)

    print("create train_loader and test_loader ...")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    print("create drug_graphs_dict and target_graphs_dict ...")

    drug_x = np.load(f'../data/{dataset}/post_generation/drugs_embedding.npy')
    target_x = np.load(f'../data/{dataset}/post_generation/targets_embedding.npy')
    drug_simName = f'{FLAGS.path}drug_{FLAGS.drug_sg}nng_{FLAGS.drug_Sname}.pt'
    target_simName = f'{FLAGS.path}target_{FLAGS.target_sg}nng_{FLAGS.target_Sname}.pt'

    drug_similarity_graph = torch.load(f'data/{dataset}/{drug_simName}')
    x, edge, weight = drug_similarity_graph
    drug_similarity_graph = torch.tensor(drug_x).to(device).float(), edge.to(device).T, weight.to(device)
    target_similarity_graph = torch.load(f'data/{dataset}/{target_simName}')
    x, edge, weight = target_similarity_graph
    target_similarity_graph = torch.tensor(target_x).to(device).float(), edge.to(device).T, weight.to(device)

    drug_graphs_dict = torch.load(f'../data/{dataset}/post_generation/drug_graph.pt')
    target_graphs_dict = torch.load(f'../data/{dataset}/post_generation/target_graph.pt')


    if FLAGS.pre == 1:
        drug_seq_embedding = np.load(f'../data/{dataset}/post_generation/drugs_embedding.npy')
        target_seq_embedding = np.load(f'../data/{dataset}/post_generation/targets_embedding.npy')
    else:
        drug_seq_embedding = np.load(f'../data/{dataset}/post_generation/drugs_embedding_label.npy')
        target_seq_embedding = np.load(f'../data/{dataset}/post_generation/targets_embedding_label.npy')


    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug", seq=drug_seq_embedding)
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=len(drug_graphs_dict))

    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target", seq=target_seq_embedding)
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=len(target_graphs_dict))

    if FLAGS.model < 6:
        architecture = Architecture()
    else:
        architecture = Architecture(layers = FLAGS.layers, pre = FLAGS.pre)
    architecture.to(device)
    print(architecture)


    if FLAGS.test==1:
        if FLAGS.model < 6:
            path_a = f"models/{dataset}/{model_name}.pt"
        else:
            path_a = f"models/{dataset}/{model_name}_{FLAGS.layers}.pt"
        architecture.load_state_dict(torch.load(path_a, map_location=device))
        architecture.eval()
        G, P =  predicting(architecture, device, test_loader, drug_graphs_DataLoader,target_graphs_DataLoader, drug_similarity_graph, target_similarity_graph)
        # np.save(f'result_data/{dataset}/DeepGIN_3.npy', P)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
        assert 0

    if fold == -100:
        best_result = [1000]

    best_epoch_mse = 0

    print("start training ...")

    for epoch in range(NUM_EPOCHS):
        train(architecture, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, LR, epoch + 1, TRAIN_BATCH_SIZE, 
              drug_similarity_graph, target_similarity_graph)
        G, P = predicting(architecture, device, test_loader, drug_graphs_DataLoader,target_graphs_DataLoader, drug_similarity_graph, target_similarity_graph)
        result = model_evaluate(G, P, dataset)
        print(result)

        if fold == -100:
            if result[0] < best_result[0]:
                best_result = result
                best_epoch_mse = epoch
                print('mse improved at epoch ', best_epoch_mse, '; best_valid_mse', best_result[0])
                checkpoint_path = f"models/{dataset}/{model_name}.pt"
                torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)
            else:
                print('No mse improved since epoch ', best_epoch_mse, '; best_valid_mse', best_result[0])


    if fold == -100:
        print('\npredicting for test data')

        G, P = predicting(architecture, device, test_loader, drug_graphs_DataLoader,target_graphs_DataLoader, drug_similarity_graph, target_similarity_graph)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
    else:
        print(f"\nbest result for fold {fold} of cross validation:")
        print("reslut:", best_result)


if __name__ == '__main__':
    seed = 1
    setup_seed(seed)
    train_test()