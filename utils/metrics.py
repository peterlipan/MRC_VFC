import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from imblearn.metrics import sensitivity_score, specificity_score


def compute_avg_metrics(groundTruth, activations):
    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    mean_acc = accuracy_score(y_true=groundTruth, y_pred=predictions)
    f1_macro = f1_score(y_true=groundTruth, y_pred=predictions, average='macro')
    try:
        auc = roc_auc_score(y_true=groundTruth, y_score=activations, multi_class='ovr')
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    bac = balanced_accuracy_score(y_true=groundTruth, y_pred=predictions)
    sens_macro = sensitivity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    spec_macro = specificity_score(y_true=groundTruth, y_pred=predictions, average='macro')

    return mean_acc, f1_macro, auc, bac, sens_macro, spec_macro


def compute_confusion_matrix(groundTruth, activations, labels):

    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    cm = confusion_matrix(y_true=groundTruth, y_pred=predictions, labels=labels)

    return cm


def epochVal(model, dataLoader):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        for i, (image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            output = model(image)
            if isinstance(output, tuple):
                _, output = output
            output = F.softmax(output, dim=1)
            groundTruth = torch.cat((groundTruth, label))
            activations = torch.cat((activations, output))

        acc, f1, auc, bac, sens, spec = compute_avg_metrics(groundTruth, activations)

    model.train(training)

    return acc, f1, auc, bac, sens, spec


def epochTest(model, dataLoader):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        for i, (image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            output = model(image)
            if isinstance(output, tuple):
                _, output = output
            output = F.softmax(output, dim=1)
            groundTruth = torch.cat((groundTruth, label))
            activations = torch.cat((activations, output))

    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    cm = confusion_matrix(y_true=groundTruth, y_pred=predictions)
    model.train(training)

    return cm
