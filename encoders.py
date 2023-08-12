from transformers import DistilBertModel, DistilBertConfig
from torch import nn
import timm

class MelSpectrogramEncoder(nn.Module):
    def __init__(self, device, model_name="resnet50"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg").to(device)
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
class TextEncoder(nn.Module):
    def __init__(self, device, model_name="distilbert-base-uncased"):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name).to(device)
            
        for p in self.model.parameters():
            p.requires_grad = True

        # Use the CLS (start) token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]