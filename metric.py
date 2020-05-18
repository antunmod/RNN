import torch


def binary_to_one_hot(binary_logits):
    """This method converts array of binary logits to one_hot. Value 0 is decision bound:
        - values < 0 -> class 0
        - values > 0 -> class 1
    """
    one_hot = torch.zeros((len(binary_logits), 2))
    for i in range(len(binary_logits)):
        one_hot[i][int(binary_logits[i] >= 0)] = 1

    return one_hot


def predict(binary_logits):
    """ This method predicts value according to the given logits"""
    return torch.Tensor([int(binary_logit > 0) for binary_logit in binary_logits])


def accuracy_f1_confusion_matrix(logits, y):
    """ This method calculates the number of correctly predicted outputs"""
    predicted = predict(logits)
    equality = torch.BoolTensor(predicted == y)
    positivity = torch.BoolTensor(y == torch.ones(len(y)))

    tp = torch.sum(equality & positivity).float()
    fp = torch.sum(~equality & ~positivity).float()
    tn = torch.sum(equality & ~positivity).float()
    fn = torch.sum(~equality & positivity).float()

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    confusion_matrix = torch.Tensor([[tp, fp], [fn, tn]])
    return accuracy, f1, confusion_matrix
