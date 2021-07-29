import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

original_model = models.densenet169(pretrained=False)
summary(original_model, input_size=(16,3,640,480),row_settings=["var_names"])
