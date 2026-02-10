import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vgg19,
    VGG19_Weights,
    alexnet,
    AlexNet_Weights,
)

class VGG19WithFeatures(nn.Module):
    """
    VGG19 with scale-variant features
    Architecture similar to ResNet50WithRT but using VGG19 backbone
    """
    def __init__(self, rt_dim=8, num_classes=4, pretrained=True):
        super(VGG19WithFeatures, self).__init__()
        
        # 1x1 Conv to transform 2 channels (VV, VH) to 3 channels (RGB)
        self.channel_adapter = nn.Conv2d(2, 3, kernel_size=1, stride=1, padding=0)
        
        # Load pretrained VGG19
        if pretrained:
            weights = VGG19_Weights.IMAGENET1K_V1
            self.vgg19 = vgg19(weights=weights)
        else:
            self.vgg19 = vgg19(weights=None)
        
        # Remove final classification layer
        self.vgg19 = nn.Sequential(*list(self.vgg19.children())[:-1])
        
        # Image feature dimension from VGG19
        vgg_out_features = 512 * 7 * 7  # After flattening
        
        # 3-layer dense network for scale-variant features
        self.feature_net = nn.Sequential(
            nn.Linear(rt_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined feature dimension
        combined_features = vgg_out_features + 16
        
        # Final classification layers with L1 regularization
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, rt):
        """
        Forward pass
        
        Args:
            image: (B, 2, 64, 64) - VV and VH stacked
            rt: (B, 8) - rotation/translation features
        
        Returns:
            logits: (B, num_classes)
        """
        # Convert 2 channels to 3 channels
        x = self.channel_adapter(image)
        
        # Extract image features with VGG19
        x = self.vgg19(x)
        x = torch.flatten(x, 1)  # (B, 512*7*7)
        
        # Process scale-variant features
        f = self.feature_net(rt)  # (B, 16)
        
        # Concatenate features
        combined = torch.cat([x, f], dim=1)  # (B, 512*7*7 + 16)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits

class AlexNetWithFeatures(nn.Module):
    """
    AlexNet with scale-variant features
    Architecture similar to ResNet50WithFeatures but using AlexNet backbone
    """
    def __init__(self, rt_dim=8, num_classes=4, pretrained=True):
        super(AlexNetWithFeatures, self).__init__()
        
        # 1x1 Conv to transform 2 channels (VV, VH) to 3 channels (RGB)
        self.channel_adapter = nn.Conv2d(2, 3, kernel_size=1, stride=1, padding=0)
        
        # Load pretrained AlexNet
        if pretrained:
            weights = AlexNet_Weights.IMAGENET1K_V1
            alexnet_model = alexnet(weights=weights)
        else:
            alexnet_model = alexnet(weights=None)
        
        # Extract only the feature extraction part (convolutional layers)
        self.features = alexnet_model.features
        
        # Extract the avgpool
        self.avgpool = alexnet_model.avgpool
        
        # Extract classifier layers except the last one
        self.alexnet_classifier = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
        
        # Image feature dimension from AlexNet (after the second-to-last FC layer)
        alexnet_out_features = 4096 
        
        # 3-layer dense network for scale-variant features
        self.feature_net = nn.Sequential(
            nn.Linear(rt_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined feature dimension
        combined_features = alexnet_out_features + 16
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, rt):
        """
        Forward pass
        
        Args:
            image: (B, 2, 64, 64) - VV and VH stacked
            rt: (B, 8) - rotation/translation features
        
        Returns:
            logits: (B, num_classes)
        """
        # Convert 2 channels to 3 channels
        x = self.channel_adapter(image)
        
        # Extract image features with AlexNet
        x = self.features(x)  # Convolutional features
        x = self.avgpool(x)   # Adaptive average pooling
        x = torch.flatten(x, 1)  # Flatten
        x = self.alexnet_classifier(x)  # Through FC layers (except last)
        
        # Process scale-variant features
        f = self.feature_net(rt)  # (B, 16)
        
        # Concatenate features
        combined = torch.cat([x, f], dim=1)  # (B, 4096 + 16)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits

class ResNet50WithRT(nn.Module):
    """
    ResNet50 with scale-variant features
    Architecture as shown in Fig. 1 of the paper
    """
    
    def __init__(self, rt_dim=8, num_classes=4, pretrained=True):
        super(ResNet50WithRT, self).__init__()
        
        # 1x1 Conv to transform 2 channels (VV, VH) to 3 channels (RGB)
        self.channel_adapter = nn.Conv2d(2, 3, kernel_size=1, stride=1, padding=0)
        
        # Load pretrained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.resnet = resnet50(weights=weights)
        else:
            self.resnet = resnet50(weights=None)
        
        # Remove final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze early layers (fine-tune last 20 layers only)
        # for i, param in enumerate(self.resnet.parameters()):
        #     if i < len(list(self.resnet.parameters())) - 20:
        #         param.requires_grad = False
        
        # Image feature dimension from ResNet50
        resnet_out_features = 2048
        
        # 3-layer dense network for scale-variant features
        self.feature_net = nn.Sequential(
            nn.Linear(rt_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined feature dimension
        combined_features = resnet_out_features + 16
        
        # Final classification layers with L1 regularization
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, image, rt):
        """
        Forward pass
        
        Args:
            image: (B, 2, 64, 64) - VV and VH stacked
            rt: (B, 8) - rotation/translation features
        
        Returns:
            logits: (B, num_classes)
        """
        # Convert 2 channels to 3 channels
        x = self.channel_adapter(image)
        
        # Extract image features with ResNet50
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # (B, 2048)
        
        # Process scale-variant features
        f = self.feature_net(rt)  # (B, 16)
        
        # Concatenate features
        combined = torch.cat([x, f], dim=1)  # (B, 2048 + 16)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits


class BaselineModel(nn.Module):
    """
    Baseline model from the paper (8 conv layers)
    """
    
    def __init__(self, rt_dim=8, num_classes=4):
        super(BaselineModel, self).__init__()
        
        # Convolutional block (8 layers with 3 max pooling)
        self.conv_block = nn.Sequential(
            # Conv block 1
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened size: 64x64 -> 8x8 after 3 pooling layers
        conv_out_size = 256 * 8 * 8
        
        # Feature network
        self.feature_net = nn.Sequential(
            nn.Linear(rt_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_size + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, image, rt):
        # Process image
        x = self.conv_block(image)
        x = torch.flatten(x, 1)
        
        # Process features
        f = self.feature_net(rt)
        
        # Combine and classify
        combined = torch.cat([x, f], dim=1)
        logits = self.classifier(combined)
        
        return logits
