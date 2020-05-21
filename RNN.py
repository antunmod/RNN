import torch
from torch import nn
import metric
import util
import baseline
import argparse


class RecurrentModel(nn.Module):

    def __init__(self, input_width, rnn1_width, rnn2_width, fc1_width, fc2_width, output_width):
        super().__init__()
        self.rnn1 = nn.RNN(input_width, rnn1_width, 2)
        self.rnn2 = nn.RNN(rnn1_width, rnn2_width, 2)
        self.fc1 = nn.Linear(fc1_width, fc2_width, bias=True)
        self.fc2 = nn.Linear(fc2_width, output_width, bias=True)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(out1, h1)
        h = self.fc1(out2)
        h = torch.relu(h)
        return self.fc2(h)[-1]  # consider only the final output


def train(model, data, optimizer, criterion, args):
    model.train()
    for batch_num, batch in enumerate(data):
        x, y, lengths = batch
        model.zero_grad()

        x_embedding = to_embedding(x)
        logits = model.forward(x_embedding).reshape(len(x))

        y_reshaped = torch.tensor(y).reshape(len(x)).type_as(logits)
        loss = criterion(logits, y_reshaped)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def evaluate(model, data, criterion):
    model.eval()
    all_logits = torch.FloatTensor()
    all_y = torch.FloatTensor()
    total_epoch_loss = 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, y, lengths = batch
            model.zero_grad()
            x_embedding = to_embedding(x)
            logits = model.forward(x_embedding).reshape(len(x))

            y_reshaped = torch.tensor(y).reshape(len(x)).type_as(logits)
            loss = criterion(logits, y_reshaped)

            # saving logits for metrics calculations
            all_logits = torch.cat((all_logits, logits))
            all_y = torch.cat((all_y, torch.FloatTensor(y)))
            total_epoch_loss += loss.data

    accuracy, f1, confusion_matrix = metric.accuracy_f1_confusion_matrix(all_logits, all_y)
    print(f'Average validate batch loss: {total_epoch_loss / (batch_num + 1)}')
    print(f'Valid accuracy: {accuracy}')
    print(f'F1: {f1}')
    print(f'Confusion matrix:\n {confusion_matrix}')


def to_embedding(batch):
    """
    This method switches vocabulary indexes with embedded vectors
    :param batch: batch of numerikalized instances
    :return: batch of
    """
    global embedding
    batch_size, instance_length, vector_dimension = batch.shape[0], batch.shape[1], embedding.weight.shape[1]
    embedding_matrix_batch = torch.zeros((batch_size, instance_length, vector_dimension))
    for instance_index in range(batch_size):
        for index in range(instance_length):
            embedding_matrix_batch[instance_index][index] = embedding.weight[batch[instance_index][index]]

    return embedding_matrix_batch


embedding = None


def main(args):
    seed = args.seed
    torch.manual_seed(seed)

    train_dataset, train_dataloader, validate_dataset, validate_dataloader, test_dataset, test_dataloader = baseline.load_datasets_and_dataloaders()

    embedding_matrix = util.embedding(train_dataset.text_vocab, util.config["glove_file_path"])
    use_freeze = util.config["glove_file_path"] is not None
    global embedding
    embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=use_freeze)

    # setup net
    input_width = 300
    rnn1_width = 150
    rnn2_width = 150
    fc1_width = 150
    fc2_width = 150
    output_width = 1

    model = RecurrentModel(input_width, rnn1_width, rnn2_width, fc1_width, fc2_width, output_width)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        print(f'----------------------------\nEpoch: {epoch}')
        train(model, train_dataloader, optimizer, criterion, args)
        evaluate(model, validate_dataloader, criterion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-epochs", type=int, required=True)
    parser.add_argument("-clip", type=float, required=True)

    args = parser.parse_args()
    main(args)
