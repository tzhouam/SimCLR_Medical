import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                            "resnet101": models.resnet101(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        print("dim_mlp:", dim_mlp)

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

        print(f"Shape of the last layer's weights: {self.backbone.fc[-1].weight.shape}")
        print(f"Shape of the last layer's biases: {self.backbone.fc[-1].bias.shape}")
        # input("Press Enter to continue...")

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid ResNet backbone architecture. Check the config file and pass one of: resnet18, resnet50, or resnet101")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class DenseNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(DenseNetSimCLR, self).__init__()
        self.densenet_dict = {"densenet121": models.densenet121(pretrained=False, num_classes=out_dim),
                            "densenet169": models.densenet169(pretrained=False, num_classes=out_dim),
                            "densenet201": models.densenet201(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.classifier.in_features
        print("dim_mlp:", dim_mlp)

        # add mlp projection head
        self.backbone.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.classifier)

        print(f"Shape of the last layer's weights: {self.backbone.classifier[-1].weight.shape}")
        print(f"Shape of the last layer's biases: {self.backbone.classifier[-1].bias.shape}")
        # input("Press Enter to continue...")

    def _get_basemodel(self, model_name):
        try:
            model = self.densenet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid DenseNet backbone architecture. Check the config file and pass one of: densenet121, densenet169, or densenet201")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
