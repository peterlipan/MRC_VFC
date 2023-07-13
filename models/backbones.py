# pre-trained model of MobileNetV2
import torchvision
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class CreateModel(nn.Module):
    def __init__(self, backbone="resnet18", ema=False, out_features=7, pretrained=False):
        super(CreateModel, self).__init__()
        models = {
            "resnet18": torchvision.models.resnet18(pretrained=pretrained),
            "resnet34": torchvision.models.resnet34(pretrained=pretrained),
            "resnet50": torchvision.models.resnet50(pretrained=pretrained),
            "resnet101": torchvision.models.resnet101(pretrained=pretrained),
            "resnet152": torchvision.models.resnet152(pretrained=pretrained),
            "densenet121": torchvision.models.densenet121(pretrained=pretrained),
            "densenet201": torchvision.models.densenet201(pretrained=pretrained),
            "efficientnet_v2_s": torchvision.models.efficientnet_v2_s(pretrained=pretrained),
            "efficientnet_v2_m": torchvision.models.efficientnet_v2_m(pretrained=pretrained),
            "efficientnet_v2_l": torchvision.models.efficientnet_v2_l(pretrained=pretrained),

        }
        assert backbone in models.keys()
        model = models[backbone]
        self.n_classes = out_features
        
        if backbone.startswith('resnet'):
            self.n_features = model.fc.in_features
            model.fc = Identity()

        elif backbone.startswith('densenet'):
            self.n_features = model.classifier.in_features
            model.classifier = Identity()
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

        elif backbone.startswith('efficient'):
            self.n_features = model.classifier[1].in_features
            model.classifier[1] = Identity()

        classifier = nn.Linear(self.n_features, out_features, bias=True)
        
        if ema:
            for param in model.parameters():
                param.detach_()
            for param in classifier.parameters():
                param.detach_()

        self.encoder = model
        self.classifier = classifier

    def forward(self, x):

        features = self.encoder(x)
        out = self.classifier(features)

        return features, out
