from tqdm import tqdm
import itertools
import torch
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from pathlib import Path

from model import AudioTextModel
from dataset import MusicCapsDataset

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def create_dataset(wav_dir, csv, batch_size):

    dataset = MusicCapsDataset(wav_dir, csv)
    
    # train_dataset = dataset[:int(len(dataset) * 0.9)]
    # validation_dataset = dataset[int(len(dataset) * 0.9):]
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    # validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
    
    train_indices = range(int(len(dataset) * 0.9))
    val_indices = range(int(len(dataset) * 0.9), len(dataset))

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=val_sampler)
    
    return train_loader, validation_loader  

def train_epoch(model, train_loader, optimizer, lr_scheduler, device='cuda'):
    losses = []
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k:v.to(device) for k, v in batch.items() if k != "fname"}
    
        batch["text"]["input_ids"] = torch.squeeze(batch["text"]["input_ids"])
        batch["text"]["attention_mask"] = torch.squeeze(batch["text"]["attention_mask"])
            
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch["image"].size(0)
        losses = losses + [loss.item() for i in range(batch_size)]

        tqdm_object.set_postfix(train_loss=np.mean(np.array(losses)), lr=get_lr(optimizer))
    return np.mean(np.array(losses))

def valid_epoch(model, valid_loader, device='cuda'):
    losses = []

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "fname"}
        
        batch["text"]["input_ids"] = torch.squeeze(batch["text"]["input_ids"])
        batch["text"]["attention_mask"] = torch.squeeze(batch["text"]["attention_mask"])
        
        loss = model(batch)

        batch_size = batch["image"].size(0)
        losses = losses + [loss.item() for i in range(batch_size)]

        tqdm_object.set_postfix(valid_loss=np.mean(np.array(losses)))
    return np.mean(np.array(losses))

def train(data_dir, data_csv, log_dir, epochs, batch_size, checkpoint, validate_interval, device):
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    model = AudioTextModel(device)
    model.to(device)
    params = [
        {"params": model.mel_spec_encoder.parameters(), "lr": 1e-4},
        {"params": model.text_encoder.parameters(), "lr": 1e-5},
        {"params": itertools.chain(
            model.mel_spec_projection.parameters(), model.text_projection.parameters()
        ), "lr": 1e-3, "weight_decay": 1e-3}
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1, factor=0.8
    )

    train_loader, validation_loader = create_dataset(data_dir, data_csv, batch_size)
    
    start_epoch = 0
    start_loss = float('inf')
    if checkpoint is not None:
        checkpoint_dict = torch.load(checkpoint)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        start_epoch = checkpoint_dict['epoch']
        start_loss = checkpoint_dict['loss']
        
        print ("Loaded model from epoch", start_epoch)

    best_loss = start_loss
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, device)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}")

        if epoch % validate_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_loss = valid_epoch(model, validation_loader)
            model.train()
            
            print(f"Epoch: {epoch}, Validation Loss: {valid_loss}")
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, f"{log_dir}/epoch-{epoch}.pt")
                print("Saved Best Model!")
        
        lr_scheduler.step(train_loss)
        # lr_scheduler.step(valid_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--data_csv")
    parser.add_argument("--log_dir", default="outputs")
    parser.add_argument("--epochs", default=40)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--validate_interval", default=5)
    parser.add_argument("--device", default='cuda')
    
    args = parser.parse_args()
    
    
    train(args.data_dir, args.data_csv, args.log_dir, int(args.epochs), int(args.batch_size), args.checkpoint, int(args.validate_interval), args.device) 