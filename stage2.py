import os
import wandb
import argparse
import torch
import numpy as np
import pandas as pd
from models import CreateModel, Linear
from data import Transforms, ISICDataset, virtual_feature_compensation
from utils.yaml_config_hook import yaml_config_hook
from torch.utils.data import DataLoader
from utils import epochVal, epochTest
from utils.loss import GCELoss


def inference(loader, backbone, device):
    feature_vector = []
    labels_vector = []
    backbone.eval()
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            activations, _ = backbone(x)

        activations = activations.detach()

        feature_vector.extend(activations.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(backbone, train_loader, test_loader, val_loader, device):
    train_X, train_y = inference(train_loader, backbone, device)
    test_X, test_y = inference(test_loader, backbone, device)
    val_X, val_y = inference(val_loader, backbone, device)
    return train_X, train_y, test_X, test_y, val_X, val_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, X_val, y_val, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    val = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, val_loader


def e_step(backbone, classifier, opt, loader, loss_func, logger):
    """
    Freeze the classifier and train the backbone,
    i.e., estimate the expected distribution of the features.
    :return:
    """
    backbone.train()
    classifier.eval()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(args.device), y.to(args.device)
        activations, _ = backbone(x)
        with torch.no_grad():
            out = classifier(activations)
        
        loss = loss_func(out, y)

        opt.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        opt.step()

        if logger is not None:
            logger.log({"E Step loss": loss.item()})


def m_step(classifier, opt, loader, loss_func, logger):
    """
    Freeze the backbone and train the classifier with virtual samples,
    i.e., maximize the expectation of the distribution of the features
    :return:
    """
    epoch_loss = 0
    epoch_acc = 0
    classifier.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(args.device), y.to(args.device)
        out = classifier(x)
        loss = loss_func(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        predict = out.argmax(1)
        acc = (predict == y).sum().item() / y.size(0)

        epoch_acc += acc
        epoch_loss += loss.item()
        if logger is not None:
            logger.log({"M Step loss": loss.item()})
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/configs.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.debug:
        wandb.login(key="[Your wandb key here]")

        config = dict()
        for k, v in yaml_config.items():
            config[k] = v

        wandb_logger = wandb.init(
            project="MRC_VFC_on_%s"%args.dataset,
            notes="MICCAI 2023",
            tags=["MICCAI23", "Class imbalance", "Dermoscopy", "Representation Learning"],
            config=config
        )
    else:
        wandb_logger = None

    transforms = Transforms(size=args.image_size)
    train_dataset = ISICDataset(args.data_path, args.csv_file_train, transform=transforms.test_transform)
    test_dataset = ISICDataset(args.data_path, args.csv_file_test, transform=transforms.test_transform)
    val_dataset = ISICDataset(args.data_path, args.csv_file_val, transform=transforms.test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # load pre-trained the backbone from checkpoint
    n_classes = train_dataset.n_class
    backbone_model = CreateModel(backbone=args.backbone, out_features=n_classes)
    model_fp = os.path.join(args.checkpoints, "epoch_{}_.pth".format(args.epochs))
    checkpoint = torch.load(model_fp, map_location=args.device)
    backbone_model.load_state_dict(checkpoint)
    backbone_model = backbone_model.to(args.device)
    backbone_optimizer = torch.optim.SGD(backbone_model.parameters(),
                                         lr=args.backbone_lr, momentum=0.9, weight_decay=1e-4)
    backbone_criterion = GCELoss(num_classes=n_classes)

    # Classifier
    classifier_model = Linear(backbone_model.n_features, backbone_model.n_classes)
    classifier_model = classifier_model.to(args.device)
    classifier_optimizer = torch.optim.SGD(classifier_model.parameters(),
                                           lr=args.classifier_lr, momentum=0.9, weight_decay=1e-4)
    classifier_criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.stage2_epochs):
        # extract features with the backbone
        train_X, train_y, test_X, test_y, val_X, val_y = get_features(
            backbone_model, train_loader, test_loader, val_loader, args.device
        )

        # Virtual sample compensation
        if args.virtual_size > 0:
            train_X, train_y = virtual_feature_compensation(train_X, train_y, n_classes, args.virtual_size)

        arr_train_loader, arr_test_loader, arr_val_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, val_X, val_y, args.stage2_batch_size
        )

        # m-step
        # the first e-step is done at the stage1
        # so, we start with m-step
        loss_epoch, acc_epoch = \
            m_step(classifier_model, classifier_optimizer, arr_train_loader, classifier_criterion, wandb_logger)

        # e-step
        e_step(backbone_model, classifier_model, backbone_optimizer, train_loader, backbone_criterion, wandb_logger)

        test_acc, test_f1, test_auc, test_bac, test_sens, test_spec = epochVal(classifier_model, arr_test_loader)
        val_acc, val_f1, val_auc, val_bac, val_sens, val_spec = epochVal(classifier_model, arr_val_loader)
        if args.wandb:
            wandb_logger.log({'test': {'Accuracy': test_acc,
                                       'F1 score': test_f1,
                                       'AUC': test_auc,
                                       'Balanced Accuracy': test_bac,
                                       'Sensitivity': test_sens,
                                       'Specificity': test_spec},
                              'validation': {'Accuracy': val_acc,
                                             'F1 score': val_f1,
                                             'AUC': val_auc,
                                             'Balanced Accuracy': val_bac,
                                             'Sensitivity': val_sens,
                                             'Specificity': val_spec}})
        print(
            f"Epoch [{epoch}/{args.stage2_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {acc_epoch / len(arr_train_loader)}"
        )
