import matplotlib.pyplot as plt
import math
import cv2
import argparse
from transformers import DistilBertTokenizer
import torch
from tqdm import tqdm
import torch.nn.functional as F
import os

from model import AudioTextModel
from dataset import CustomMusicDataset, MusicCapsDataset

def get_query_embeddings(model, query, device):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    encoded_query = tokenizer(query, padding='max_length', truncation=False, max_length=200, return_tensors='pt')
    text_batch = {key: v.to(device) for key, v in encoded_query.items()}
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=text_batch["input_ids"], attention_mask=text_batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
        
    return text_embeddings

# def process_dataset(model, data_dir, csv, device, batch_size=32):
#     dataset = MusicCapsDataset(data_dir, csv)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, )
#     valid_image_embeddings = []
#     fnames = []
#     with torch.no_grad():
#         for batch in tqdm(dataloader):
#             image_features = model.mel_spec_encoder(batch["image"].to(device))
#             image_embeddings = model.mel_spec_projection(image_features)
#             valid_image_embeddings.append(image_embeddings)
#             fnames = fnames + batch["fname"]
#     return torch.cat(valid_image_embeddings), fnames

def process_dataset(model, data_dir, device, batch_size=64):
    dataset = CustomMusicDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    valid_image_embeddings = []
    fnames = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_features = model.mel_spec_encoder(batch["image"].to(device))
            image_embeddings = model.mel_spec_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
            fnames = fnames + batch["fname"]
    return torch.cat(valid_image_embeddings), fnames
    
def inference(model_path, data_dir, n, device):
    model = AudioTextModel(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print (checkpoint['loss'])
    model.eval()
    
    image_embeddings, fnames = process_dataset(model, data_dir, device)
    
    while True:
        
        query = input("Input query: ")
    
        text_embeddings = get_query_embeddings(model, query, device)

        image_embeddings_normalized = F.normalize(image_embeddings, dim=-1)
        text_embeddings_normalized = F.normalize(text_embeddings, dim=-1)
        dot_similarity = text_embeddings_normalized @ image_embeddings_normalized.T

        values, indices = torch.topk(dot_similarity.squeeze(0), n)
        matches = [fnames[idx] for idx in indices]
        
        print()
        for idx, match in enumerate(matches):
            print (f"{idx+1}. {match}")
    
#     _, axes = plt.subplots(int(math.sqrt(n)), int(math.sqrt(n)), figsize=(10, 10))
#     for match, ax in zip(matches, axes.flatten()):
#         print (match)
#         # image = cv2.imread(f"{CFG.image_path}/{match}")
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # ax.imshow(image)
#         ax.axis("off")
    
#     plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    # parser.add_argument("--query")
    parser.add_argument("--data_dir")
    # parser.add_argument("--data_csv")
    parser.add_argument("--num_matches", default=5)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    inference(args.model_path, args.data_dir, int(args.num_matches), device)