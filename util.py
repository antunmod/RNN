from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Instance:
    """Simple data wrapper"""
    text: str
    label: str


class NLPDataset(Dataset):
    """A custom dataset"""

    def __init__(self):
        self.instances = []
        self.text_vocab = None
        self.label_vocab = None

    def __getitem__(self, instances_range):
        instances_in_range = self.instances[instances_range]

        # if only a single element is requested, return elements as a tuple
        if type(instances_in_range) == Instance and instances_in_range:
            return self.text_vocab.encode(instances_in_range.text), self.label_vocab.encode(instances_in_range.label)

        # otherwise return elements in a list
        encoded_instances = []
        for instance in instances_in_range:
            text_encoded = self.text_vocab.encode(instance.text)
            label_encoded = self.label_vocab.encode(instance.label)
            encoded_instances.append((text_encoded, label_encoded))

        return encoded_instances

    def from_file(self, file_path):
        file = open(file_path, "r")
        lines = file.readlines()
        # for value in data.values:
        # if "\\" in value[0]:
        #     value[0] = value[0].replace("\\", "")
        # if "\'" in value[0]:
        #     value[0] = value[0].replace("\'", "'")
        for line in lines:
            parts = line.split(", ")
            self.instances.append(Instance(parts[0], parts[1].strip()))

        # create vocabs
        text_frequencies, label_frequencies = frequencies(self.instances)
        self.text_vocab = Vocab(text_frequencies)
        self.label_vocab = Vocab(label_frequencies)

        return self

    def __len__(self):
        return len(self.instances)


class Vocab:
    """A class for numericalization"""

    def __init__(self, frequencies, max_size=-1, min_freq=1):
        self.max_size = max_size
        self.min_freq = min_freq

        # construct itos
        self.itos = dict()
        for word in frequencies:
            frequency = frequencies[word]
            # if max size or min frequency limit reached, stop adding to dictionary
            if (self.max_size != -1 and len(self.itos) > self.max_size) or (frequency < self.min_freq):
                break
            self.itos[len(self.itos)] = word

        self.stoi = {v: k for k, v in self.itos.items()}

    def encode(self, text):
        # return single tensor value if the text is a single word
        if " " not in text and (text == "positive" or text == "negative"):
            return torch.tensor(self.stoi[text])

        # otherwise return tensor array
        tokens = text.split()
        indexes = []

        for token in tokens:
            indexes.append(self.stoi[token])

        return torch.tensor(indexes)


def pad_collate_fn(batch, pad_index=0):
    """
    Arguments:
      Batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
      A tensor representing the input batch.
    """

    texts, labels = zip(*batch)  # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts])  # Needed for later
    # Process the text instances
    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    return texts, labels, lengths


def frequencies(instances):
    text_frequencies = dict()
    label_frequencies = dict()

    # add padding and unknown tokens
    text_frequencies["<PAD>"] = 100001
    text_frequencies["<UNK>"] = 100000

    for instance in instances:
        for word in instance.text.split():
            if word in text_frequencies:
                text_frequencies[word] = text_frequencies[word] + 1
            else:
                text_frequencies[word] = 1
        if instance.label in text_frequencies:
            label_frequencies[instance.label] = label_frequencies[instance.label] + 1
        else:
            label_frequencies[instance.label] = 1

    text_frequencies = {k: v for k, v in reversed(sorted(text_frequencies.items(), key=lambda item: item[1]))}
    label_frequencies = {k: v for k, v in reversed(sorted(label_frequencies.items(), key=lambda item: item[1]))}

    return text_frequencies, label_frequencies


def embedding(vocab, file_path=None):
    # initialize all values with normal distribution
    matrix = torch.randn(len(vocab.itos), 300)

    # initialize padding values with zeros
    matrix[0] = torch.zeros(300)

    # if the glove file was not provided, return
    if file_path is None:
        return matrix

    file = open(file_path, "r")
    lines = file.readlines()

    for line in lines:
        word_values_split = line.split(" ", 1)
        word = word_values_split[0]
        values = word_values_split[1].split()
        values_float = [float(value) for value in values]

        # initialize word index to 1 in case it is not in the vocabulary
        index = 1
        if word in vocab.stoi:
            index = vocab.stoi[word]

        matrix[index] = torch.tensor(values_float)

    return matrix


def initialize_dataset(file_path):
    nlp_dataset = NLPDataset()
    return nlp_dataset.from_file(file_path=file_path)


def initialize_dataset_and_dataloader(file_path, batch_size):
    dataset = initialize_dataset(file_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=config["shuffle"], collate_fn=pad_collate_fn)
    return dataset, dataloader


def initialize_dataloader(file_path, batch_size):
    dataset, dataloader = initialize_dataset_and_dataloader(file_path, batch_size)
    return dataloader


config = {}
config["test_data_file_path"] = 'data/sst_test_raw.csv'
config["shuffle"] = True
config["train_data_file_path"] = 'data/sst_train_raw.csv'
config["validate_data_file_path"] = 'data/sst_valid_raw.csv'
config["glove_file_path"] = 'data/sst_glove_6b_300d.txt'  # should be set to None if it is not used

if __name__ == "__main__":
    config["batch_size"] = 10

    dataset, train_dataloader = initialize_dataset_and_dataloader(config["train_data_file_path"])

    texts, labels, lengths = next(iter(train_dataloader))
    print(dataset[0])

    # create temporary frequencies and vocabs
    text_frequencies, label_frequencies = frequencies(dataset[:])
    text_vocab = Vocab(text_frequencies)
    label_vocab = Vocab(label_frequencies)

    embedding_matrix = embedding(text_vocab, config["glove_file_path"])
    use_freeze = config["glove_file_path"] is not None
    embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=use_freeze)
