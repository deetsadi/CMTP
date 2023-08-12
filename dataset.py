from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchaudio import transforms
from torchaudio import functional
import torchaudio
from transformers import DistilBertTokenizer
import numpy as np
import torch
import datasets
from glob import glob
from pathlib import Path
import pandas as pd
import ast
import soundfile as sf
from tqdm import tqdm
import torch
import numpy
from torch.utils.data.sampler import SubsetRandomSampler

class MusicCapsDataset(Dataset):
    def __init__(self, audio_dir, csv, y_res=128, n_fft=2048, hop_length=512, sample_rate=44100, duration=10):
        self.data = self.setup_dataset(list(Path(audio_dir).glob("*.wav")))
        self.df = pd.read_csv(csv)
        
        self.y_res = y_res
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.duration = duration
        
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.mel_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=y_res)
    
    def setup_dataset(self, data):
                
        print ("Setting up dataset...")
        
        
        new_data = []
        current_global_index = 0
        current_file_index = 0
        
        for idx, path in enumerate(tqdm(data)):
            audio_data, sample_rate = sf.read(path)
            duration = round(len(audio_data) / float(sample_rate), 2)
            
            new_data.append((data[idx], 0))
            
            if duration > 15:
                new_data.append((data[idx],1))
        
        return new_data
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fname, start_flag = self.data[index]        
        audio, sr = torchaudio.load(fname)
        resampled_audio = functional.resample(audio, sr, self.sample_rate)
        mono_audio = torch.mean(resampled_audio, dim=0).unsqueeze(0)
        
        if start_flag == 1:
            mono_audio = mono_audio[:, self.sample_rate * self.duration:]
        else:
            mono_audio = mono_audio[:, :self.sample_rate * self.duration]
            
        padded_audio = torch.cat((mono_audio, torch.zeros((1, self.sample_rate * self.duration - mono_audio.shape[1]))), dim=1)
        mel = self.mel_transform(padded_audio)
        
        ytid = str(self.data[index]).split(".")[0].split("/")[-1]
        caption = self.df.loc[self.df["ytid"] == ytid, "caption"].iloc[0]
        tokenized_caption = self.tokenizer(caption, padding='max_length', truncation=False, max_length=200, return_tensors='pt')
        
        return {
            "image" : torch.cat((mel,mel,mel)),
            "text" : tokenized_caption,
            "fname" : str(fname),
            # "caption" : caption
        }  

class CustomMusicDataset(Dataset):
    def __init__(self, audio_dir, y_res=128, n_fft=2048, hop_length=512, sample_rate=44100, duration=10):
        
        self.y_res = y_res
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.duration = duration
        
        self.data = self.setup_dataset(list(Path(audio_dir).rglob("*.wav")))
        self.mel_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=y_res)
                
    def __len__(self):
        return len(self.data)
    
    def setup_dataset(self, data):
                
        print ("Setting up dataset...")
        
        
        new_data = []
        
        for idx, path in enumerate(tqdm(data)):
            current_multiple = 0
            try:
                audio_data, sample_rate = sf.read(path)
            except:
                continue
            duration = round(len(audio_data) / float(sample_rate), 2)
            
            for i in range(int(duration // self.duration)):
            
                new_data.append((data[idx], current_multiple)) 
                current_multiple += 1
        
        return new_data
    
    def __getitem__(self, index):
        fname, start_flag = self.data[index]        
        audio, sr = torchaudio.load(fname)
        resampled_audio = functional.resample(audio, sr, self.sample_rate)
        mono_audio = torch.mean(resampled_audio, dim=0).unsqueeze(0)
        
        mono_audio = mono_audio[:, self.sample_rate * self.duration * start_flag:self.sample_rate * self.duration * (start_flag+1)]
        
        try:    
            padded_audio = torch.cat((mono_audio, torch.zeros((1, self.sample_rate * self.duration - mono_audio.shape[1]))), dim=1)
        except:
            print (fname)
        mel = self.mel_transform(padded_audio)

        return {
            "image" : torch.cat((mel,mel,mel)),
            "fname" : str(fname),
        }  