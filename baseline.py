import torch
from torch import nn
import util
import argparse
import metric
import random


class BaselineModel(nn.Module):

    def __init__(self, fc1_width, fc2_width, fc3_width, output_width):
        super().__init__()
        self.fc1 = nn.Linear(fc1_width, fc2_width, bias=True)
        self.fc2 = nn.Linear(fc2_width, fc3_width, bias=True)
        self.fc3 = nn.Linear(fc3_width, output_width, bias=True)

    def forward(self, x):
        h = torch.mean(x, 1)
        h = self.fc1(h)
        h = torch.relu(h)

        h = self.fc2(h)
        h = torch.relu(h)
        return self.fc3(h)


def train(model, data, optimizer, criterion, embedding, clip):
    model.train()
    for batch_num, batch in enumerate(data):
        x, y, lengths = batch
        model.zero_grad()

        x_embedding = to_embedding(x, embedding)
        logits = model.forward(x_embedding).reshape(len(x))

        y_reshaped = torch.tensor(y).reshape(len(x)).type_as(logits)
        loss = criterion(logits, y_reshaped)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()


def evaluate(model, data, criterion, embedding):
    model.eval()
    all_logits = torch.FloatTensor()
    all_y = torch.FloatTensor()
    average_batch_loss = 0

    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, y, lengths = batch
            model.zero_grad()
            x_embedding = to_embedding(x, embedding)
            logits = model.forward(x_embedding).reshape(len(x))

            y_reshaped = torch.tensor(y).reshape(len(x)).type_as(logits)
            loss = criterion(logits, y_reshaped)

            # saving logits for metrics calculations
            all_logits = torch.cat((all_logits, logits))
            all_y = torch.cat((all_y, torch.FloatTensor(y)))
            average_batch_loss += loss.data

    return metric.accuracy_f1_confusion_matrix(all_logits, all_y)


def load_datasets_and_dataloaders():
    train_dataset, train_dataloader = util.initialize_dataset_and_dataloader(util.config["train_data_file_path"],
                                                                             config["train_batch_size"], shuffle=True)

    validate_dataset, validate_dataloader = util.initialize_dataset_and_dataloader(
        util.config["validate_data_file_path"], config["validate_batch_size"])
    test_dataset, test_dataloader = util.initialize_dataset_and_dataloader(util.config["test_data_file_path"],
                                                                           config["validate_batch_size"])

    # create vocabs and assign them
    text_frequencies, label_frequencies = util.frequencies(train_dataset.instances)
    text_vocab = util.Vocab(text_frequencies)
    label_vocab = util.Vocab(label_frequencies)

    train_dataset.text_vocab = validate_dataset.text_vocab = test_dataset.text_vocab = text_vocab
    train_dataset.label_vocab = validate_dataset.label_vocab = test_dataset.label_vocab = label_vocab

    return train_dataset, train_dataloader, validate_dataset, validate_dataloader, test_dataset, test_dataloader


def to_embedding(batch, embedding):
    """
    This method switches vocabulary indexes with embedded vectors
    :param batch: batch of numerikalized instances
    :return: batch of
    """
    batch_size, instance_length, vector_dimension = batch.shape[0], batch.shape[1], embedding.weight.shape[1]
    embedding_matrix_batch = torch.zeros((batch_size, instance_length, vector_dimension))
    for instance_index in range(batch_size):
        for index in range(instance_length):
            embedding_matrix_batch[instance_index][index] = embedding.weight[batch[instance_index][index]]

    return embedding_matrix_batch


config = {}
config["train_batch_size"] = 10
config["validate_batch_size"] = 32


def main(args):
    train_dataset, train_dataloader, validate_dataset, validate_dataloader, test_dataset, test_dataloader = load_datasets_and_dataloaders()

    embedding_matrix = util.embedding(train_dataset.text_vocab, util.config["glove_file_path"])
    use_freeze = util.config["glove_file_path"] is not None
    embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=use_freeze)

    # setup net
    fc1_width = 300
    fc2_width = 150
    fc3_width = 150
    output_width = 1

    model = BaselineModel(fc1_width, fc2_width, fc3_width, output_width)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        print(f'----------------------------\nEpoch: {epoch}')
        train(model, train_dataloader, optimizer, criterion, embedding, args.clip)
        evaluate(model, validate_dataloader, criterion, embedding)


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
