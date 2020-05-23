import torch
from torch import nn
import metric
import util
import baseline
import argparse
import random


class RecurrentModel(nn.Module):

    def __init__(self, cell_type, input_width, hidden_size, fc1_width, fc2_width, output_width, num_layers, bidirectional=False):
        super().__init__()
        self.recurrent1 = create_layer(cell_type, input_width, hidden_size, num_layers, bidirectional)
        self.recurrent2 = create_layer(cell_type, hidden_size, hidden_size, num_layers, bidirectional)
        self.fc1 = nn.Linear(fc1_width, fc2_width, bias=True)
        self.fc2 = nn.Linear(fc2_width, output_width, bias=True)

        self.directions = 1
        if bidirectional:
            self.directions = 2

    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        x = x.permute(1, 0, 2)
        out1, h1 = self.recurrent1(x)
        out1 = out1.view(seq_length, batch_size, self.directions, self.recurrent2.hidden_size)[:, :, -1]
        # h1 = h1.view(self.recurrent1.num_layers, self.directions, batch_size, self.recurrent2.hidden_size)[-1]
        out2, h2 = self.recurrent2(out1, h1)
        out2 = out2.view(seq_length, batch_size, self.directions, self.recurrent2.hidden_size)[:, :, -1]
        h = self.fc1(out2[-1])   # consider only the final output
        h = torch.relu(h)
        return self.fc2(h)


def create_layer(cell_type, input_width, hidden_width, num_layers, bidirectional):
    if cell_type == 'RNN':
        return nn.RNN(input_width, hidden_width, num_layers, bidirectional=bidirectional)
    if cell_type == 'GRU':
        return nn.GRU(input_width, hidden_width, num_layers, bidirectional=bidirectional)
    if cell_type == 'LSTM':
        return nn.LSTM(input_width, hidden_width, num_layers, bidirectional=bidirectional)


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
    total_epoch_loss = 0
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
            total_epoch_loss += loss.data

    return metric.accuracy_f1_confusion_matrix(all_logits, all_y)


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


def main(args):
    train_dataset, train_dataloader, validate_dataset, validate_dataloader, test_dataset, test_dataloader = baseline.load_datasets_and_dataloaders()

    embedding_matrix = util.embedding(train_dataset.text_vocab, util.config["glove_file_path"])
    use_freeze = util.config["glove_file_path"] is not None
    embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=use_freeze)

    # setup net
    input_width = 300
    hidden_size = 150
    fc1_width = 150
    fc2_width = 150
    output_width = 1
    num_layers = 2

    model = RecurrentModel('LSTM', input_width, hidden_size, fc1_width, fc2_width, output_width, num_layers)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        print(f'----------------------------\nEpoch: {epoch}')
        train(model, train_dataloader, optimizer, criterion, embedding, args.clip)
        evaluate(model, validate_dataloader, criterion, embedding)
    evaluate(model, test_dataloader, criterion)


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
