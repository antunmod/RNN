import torch


def predict(binary_logits):
    """ This method predicts value according to the given logits"""
    return torch.Tensor([int(binary_logit >= 0) for binary_logit in binary_logits])


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
