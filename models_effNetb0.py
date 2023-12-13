import torch
import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



