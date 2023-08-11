# CMTP
Contrastive Music-Text Pre-Training (CMTP) is a model that has learnt relationships between music and text, enabling fully zero-shot retrieval of music based on natural language prompts and vice versa.
As an example, after training, the model was given a completely new and unlabeled set of ~1000 audio files to act as a database of known music. Next, the model was given the prompt `a slow, haunting melody`. Below are the audio files it found most closely matched this prompt:

Impressive! Lets try again with the prompt `a male singer over an electronic drumbeat`:

Inspired by CLIP from OpenAI, CMTP was trained using (music, text) pairs from the [MusicCaps](https://research.google/resources/datasets/musiccaps/) dataset. Specifically, music samples were converted to Mel-spectrograms using an emprically determined optimal set of parameters. From this point, a ResNet model is used as a 
A detailed look at the model architecture is shown below:  
![Training drawio-2](https://github.com/deetsadi/CMTP/assets/47929718/951c59dd-a6a1-426b-bee0-0425b8eea011)

Further, the contrastive loss is shown more in depth:
![Loss drawio-2](https://github.com/deetsadi/CMTP/assets/47929718/bb95fb60-842b-47a1-a338-2db40cc15cdd)

## Usage
### Dataset
Download the MusicCaps dataset, including both the raw audio files and csv containing metadata. Use the script located [here]((https://github.com/nateraw/download-musiccaps-dataset)) to do so.
### Train
Call `train.py` with your desired hyperparameters.
