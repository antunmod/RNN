import torch
from torch import nn
import util
import argparse
import metric


class BaselineModel(nn.Module):

    def __init__(self, fc1_width, fc2_width, fc3_width, output_width):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 300))
        self.fc1 = nn.Linear(fc1_width, fc2_width, bias=True)
        self.fc2 = nn.Linear(fc2_width, fc3_width, bias=True)
        self.fc3 = nn.Linear(fc3_width, output_width, bias=True)

    def forward(self, x):
        h = self.pool(x)
        h = self.fc1(h)
        h = torch.relu(h)

        h = self.fc2(h)
        h = torch.relu(h)
        return self.fc3(h)


def train(model, data, optimizer, criterion, args):
    model.train()
    for batch_num, batch in enumerate(data):
        x, y, lengths = batch
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        model.zero_grad()

        x_embedding = to_embedding(x)
        logits = model.forward(x_embedding)

        y_reshaped = torch.reshape(torch.tensor(y), (len(x), 1, 1, 1))
        loss = criterion(logits, y_reshaped.type_as(logits))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def evaluate(model, data, criterion, args):
    model.eval()
    all_logits = torch.FloatTensor()
    all_y = torch.FloatTensor()
    average_batch_loss = 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, y, lengths = batch
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
            model.zero_grad()
            x_embedding = to_embedding(x)
            logits = model.forward(x_embedding)

            y_reshaped = torch.reshape(torch.tensor(y), (len(x), 1, 1, 1))
            loss = criterion(logits,  y_reshaped.type_as(logits))

            # saving logits for metrics calculations
            all_logits = torch.cat((all_logits, torch.flatten(logits)), 0)
            all_y = torch.cat((all_y, torch.FloatTensor(y)), 0)
            average_batch_loss += loss.data

    accuracy, f1, confusion_matrix = metric.accuracy_f1_confusion_matrix(all_logits, all_y)
    print(f'Average validate batch loss: {average_batch_loss / (batch_num + 1)}')
    print(f'Valid accuracy: {accuracy}')
    print(f'F1: {f1}')
    print(f'Confusion matrix:\n {confusion_matrix}')


def load_datasets_and_dataloaders():
    train_dataset, train_dataloader = util.initialize_dataset_and_dataloader(util.config["train_data_file_path"],
                                                                             config["train_batch_size"])
    validate_dataset, validate_dataloader = util.initialize_dataset_and_dataloader(
        util.config["validate_data_file_path"], config["validate_batch_size"])
    test_dataset, test_dataloader = util.initialize_dataset_and_dataloader(util.config["test_data_file_path"],
                                                                           config["validate_batch_size"])

    return train_dataset, train_dataloader, validate_dataset, validate_dataloader, test_dataset, test_dataloader


def to_embedding(batch):
    """
    This method switches vocabulary indexes with embedded vectors
    :param batch: batch of numerikalized instances
    :return: batch of
    """
    global embedding
    batch_size, no_of_channels, instance_length, vector_dimension = batch.shape[0], batch.shape[1], batch.shape[2], embedding.weight.shape[1]
    embedding_matrix_batch = torch.zeros((batch_size, no_of_channels, instance_length, vector_dimension))
    for instance_index in range(batch_size):
        for index in range(instance_length):
            embedding_matrix_batch[instance_index][no_of_channels - 1][index] = embedding.weight[batch[instance_index][0][index]]

    return embedding_matrix_batch


config = {}
config["train_batch_size"] = 10
config["validate_batch_size"] = 32

embedding = None


def main(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = args.seed
    torch.manual_seed(seed)

    train_dataset, train_dataloader, validate_dataset, validate_dataloader, test_dataset, test_dataloader = load_datasets_and_dataloaders()

    embedding_matrix = util.embedding(train_dataset.text_vocab, util.config["glove_file_path"])
    use_freeze = util.config["glove_file_path"] is not None
    global embedding
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
        train(model, train_dataloader, optimizer, criterion, args)
        evaluate(model, validate_dataloader, criterion, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-epochs", type=int, required=True)
    parser.add_argument("-clip", type=int, required=True)

    args = parser.parse_args()
    main(args)
