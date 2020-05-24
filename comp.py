import torch
from torch import nn
import util
import RNN
import baseline
import argparse
import random
from xlwt import Workbook
import time
import os


def cell_scores_for_initial_hiperparameters(train_dataloader, validate_dataloader, test_dataloader, embedding, args):
    """This method will calculate scores for all RNN types and save the results to '1_predefined.txt'"""
    configs = []
    # setup net
    initial_config = {}
    initial_config["seed"] = args.seed
    initial_config["clip"] = args.clip
    initial_config["epochs"] = args.epochs
    initial_config["input_width"] = 300
    initial_config["hidden_size"] = 150
    initial_config["fc1_width"] = 150
    initial_config["fc2_width"] = 150
    initial_config["output_width"] = 1
    initial_config["num_layers"] = 2

    recurrent_models = ['RNN', 'GRU', 'LSTM']
    for recurrent_model in recurrent_models:
        config = {}
        config["model"] = recurrent_model
        config.update(initial_config)
        start = time.time()

        model = RNN.RecurrentModel(recurrent_model, config["input_width"], config["hidden_size"], config["fc1_width"],
                                   config["fc2_width"], config["output_width"], config["num_layers"])

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(args.epochs):
            print(f'----------------------------\nEpoch: {epoch}')
            RNN.train(model, train_dataloader, optimizer, criterion, embedding, args.clip)
            RNN.evaluate(model, validate_dataloader, criterion, embedding)
        accuracy, f1, confusion_matrix = RNN.evaluate(model, test_dataloader, criterion, embedding)
        config["accuracy"] = accuracy.item()
        config["f1"] = f1.item()
        config["TP"] = confusion_matrix[0, 0].item()
        config["FP"] = confusion_matrix[0, 1].item()
        config["FN"] = confusion_matrix[1, 0].item()
        config["TN"] = confusion_matrix[1, 1].item()

        end = time.time()
        config["time"] = end - start
        configs.append(config)

    print_to_file("1_pretrained_RNN.xls", "RNN with predefined values", configs)


def cell_scores_for_variable_hiperparameters(train_dataloader, validate_dataloader, test_dataloader, embedding, args):
    """This method will calculate scores for all RNN types and save the results to '1_predefined.txt'"""
    configs = []
    # setup net
    hyperparameters = {}
    hyperparameters["hidden_size"] = [50, 100, 10, 30, 150]
    hyperparameters["num_layers"] = [2, 3, 4, 1]
    hyperparameters["dropout"] = [0.1, 0.3, 0.7, 0.5, 0.9]
    hyperparameters["bidirectional"] = [True, False]

    values = random_search_indexes(hyperparameters)

    initial_config = {}
    initial_config["seed"] = args.seed
    initial_config["clip"] = args.clip

    net_config = {}
    net_config["input_width"] = 300
    net_config["output_width"] = 1

    recurrent_models = ['RNN', 'GRU', 'LSTM']
    print(f'Random testing with {len(recurrent_models)} models and {len(values)} values')
    for recurrent_model in recurrent_models:
        print(f'------------------------------------\n{recurrent_model}')
        for iteration, value in enumerate(values):
            print(f'{iteration}: {value}')
            config = {}
            config["model"] = recurrent_model
            config.update(initial_config)
            config.update(value)
            start = time.time()

            model = RNN.RecurrentModel(recurrent_model, net_config["input_width"], value["hidden_size"], net_config["output_width"],
                                       value["num_layers"], value["bidirectional"], value["dropout"])

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            for epoch in range(args.epochs):
                print(f'Epoch: {epoch}')
                RNN.train(model, train_dataloader, optimizer, criterion, embedding, args.clip)
                RNN.evaluate(model, validate_dataloader, criterion, embedding)
            accuracy, f1, confusion_matrix = RNN.evaluate(model, test_dataloader, criterion, embedding)
            config["accuracy"] = accuracy.item()
            config["f1"] = f1.item()
            config["TP"] = confusion_matrix[0, 0].item()
            config["FP"] = confusion_matrix[0, 1].item()
            config["FN"] = confusion_matrix[1, 0].item()
            config["TN"] = confusion_matrix[1, 1].item()

            end = time.time()
            config["time"] = end - start
            configs.append(config)

    print_to_file("2c_variable_RNN_bidirectional_sum.xls", "Random hyperparameters", configs)


def print_to_file(file_name, description, configs):
    file_path = folder_path + file_name
    if os.path.exists(file_path):
        os.remove(file_path)

    wb = Workbook()
    sheet = wb.add_sheet(description)

    config = configs[0]

    for index, key in enumerate(config):
        sheet.write(0, index, key)

    for i in range(len(configs)):
        config_values = configs[i]
        for index, (key, value) in enumerate(config_values.items()):
            sheet.write(i + 1, index, value)

    wb.save(file_path)


folder_path = 'statistics/'


def random_search_indexes(config, iterations=10):
    """This method will return a list of randomly generated config value indexes to be used when searching for best hyperparameters"""
    index_dict = {}
    for key in config.keys():
        index_dict[key] = 0

    values = []
    for i in range(iterations):
        tmp = {}
        for key in config.keys():
            # increase index randomly
            if random.random() > 0.4:
                index_dict[key] = index_dict[key] + 1
                if index_dict[key] >= len(config[key]):
                    index_dict[key] = 0

            # dropout works only when there are 2 or more layers
            if key == "dropout" and tmp["num_layers"] < 2:
                tmp[key] = 0
            else:
                tmp[key] = config[key][index_dict[key]]
        if tmp not in values:
            values.append(tmp)

    return values


def main(args):
    train_dataset, train_dataloader, validate_dataset, validate_dataloader, test_dataset, test_dataloader = baseline.load_datasets_and_dataloaders()

    embedding_matrix = util.embedding(train_dataset.text_vocab, util.config["glove_file_path"])
    use_freeze = util.config["glove_file_path"] is not None
    embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=use_freeze)

    # cell_scores_for_initial_hiperparameters(train_dataloader, validate_dataloader, test_dataloader, embedding, args)
    cell_scores_for_variable_hiperparameters(train_dataloader, validate_dataloader, test_dataloader, embedding, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int)
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-clip", type=float)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 15000)
    random.seed(args.seed)

    if args.epochs is None:
        args.epochs = 5

    if args.clip is None:
        args.clip = random.random()

    torch.manual_seed(args.seed)
    main(args)
