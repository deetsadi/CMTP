import torch
from torch import nn

import torch
from torch import nn
import torch.nn.functional as F
from encoders import MelSpectrogramEncoder, TextEncoder

class AudioTextModel(nn.Module):
    def __init__(
        self,
        device,
        temperature=1,
        mel_spec_embedding_dim=2048,
        text_embedding_dim=768,
    ):
        super().__init__()
        self.mel_spec_encoder = MelSpectrogramEncoder(device)
        self.text_encoder = TextEncoder(device)
        self.mel_spec_projection = ProjectionHead(embedding_dim=mel_spec_embedding_dim).to(device)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dim).to(device)
        self.temperature = temperature
        
    def forward(self, batch):
        # Getting Mel and Text Features
        mel_spec_features = self.mel_spec_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["text"]["input_ids"], attention_mask=batch["text"]["attention_mask"]
        )
        
        mel_spec_embeddings = self.mel_spec_projection(mel_spec_features)
        text_embeddings = self.text_projection(text_features)

        return self.contrastive_loss(mel_spec_embeddings, text_embeddings)
    
    def contrastive_loss(self, mel_spec_embeddings, text_embeddings):
        logits = (text_embeddings @ mel_spec_embeddings.T) / self.temperature # ideally would be the identity matrix, as text and mel embeddings matrices should be the same
        mel_spec_similarity = mel_spec_embeddings @ mel_spec_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (mel_spec_similarity + texts_similarity) / 2 * self.temperature, dim=-1 # identity matrix with shape (batch, batch)
        )
        texts_loss = self.cross_entropy(targets, logits) 
        mel_spec_loss = self.cross_entropy(targets.T, logits.T)
        loss =  (mel_spec_loss + texts_loss)
        return loss.mean()
    
    def cross_entropy(self, targets, preds):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        return loss
    
#     def projection(self, embedding, embedding_dim, projection_dim=256, dropout=0.1):
        
#         proj = torch.Sequential([
#             nn.Linear(embedding.shape[1], projection_dim),
#             nn.GELU(),
#             nn.Linear(projection_dim, projection_dim),
#             nn.Dropout(dropout),
#         ])
#         norm = nn.LayerNorm(projection_dim)
        
#         embedding = proj(embedding) + embedding
        
#         return norm(embedding)

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
