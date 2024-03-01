import torch.nn as nn
from transformers import ViTModel

class ViTForImageClassification(nn.Module):
    def __init__(self, num_class=2, pretrained_model_path="google/vit-base-patch16-224-in21k"):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_class)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        outputs = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(outputs)
        return logits