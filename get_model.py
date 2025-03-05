from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    densenet121, DenseNet121_Weights,
    vit_b_16, ViT_B_16_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    regnet_x_400mf, RegNet_X_400MF_Weights
)

def get_model(model_name, num_classes):
    """
    Create a model with pretrained weights and modified classifier layer
    
    Args:
        model_name (str): Name of the model to use
        num_classes (int): Number of output classes
        
    Returns:
        model: PyTorch model
    """
    if model_name == "efficientnet":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "mobilenetv3":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        
    elif model_name == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == "vit":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    elif model_name == "convnext":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        
    elif model_name == "regnet":
        weights = RegNet_X_400MF_Weights.IMAGENET1K_V1
        model = regnet_x_400mf(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model