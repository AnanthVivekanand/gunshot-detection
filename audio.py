# The backend continually records audio, preprocesses the audio
# and makes a classification. The classification is accessible 
# through an API endpoint that the frontend can access through 
# HTTP

# Ensure that SoundCard is installed: https://pypi.org/project/SoundCard/
# !python3 -m pip install SoundCard

import sounddevice as sd
import numpy as np
import threading
import torch
import torchaudio
import torchvision
import time

# print(sd.default.device)
# device_info = sd.query_devices()
# print(device_info)

duration = 4.0
fs = 22050
sd.default.samplerate = fs
duration = 4  # seconds
audio = np.zeros((fs*duration, 1))
normalize = torchvision.transforms.Normalize(mean=[-2.0064, -1.2480, -0.4483], std=[3.9764, 3.9608, 3.9427], inplace=False)

import torchvision.models as models
import torch.nn as nn

class MobileNetV3(nn.Module):
  def __init__(self):
    super(MobileNetV3, self).__init__()
    num_classes = 10
    self.model = models.mobilenet_v3_large(pretrained=True)
    self.model.classifier[3] = nn.Linear(in_features= 1280, out_features= 10)
		
  def forward(self, x):
    output = self.model(x)
    return output

model = MobileNetV3()
model.load_state_dict(torch.load("model_assets/model_2.pt", map_location=torch.device('cpu')))

#model = torch.load("drive/MyDrive/model.pt", map_location=torch.device('cpu'))
model.eval()

classifications = []

stop_threads = False

def manage_api(classification, threads):
    classifications.append(classification.squeeze().tolist())
    print(classifications)
    return

def run_model(spec, threads):
    with torch.no_grad():
        threads.append(threading.Thread(target=manage_api, args=(model(spec),threads)).start())

def preprocesses_audio(audio, threads):
    print("Preprocessing audio...")
    soundData = torch.mean(torch.from_numpy(audio), dim=1, keepdim=True)
    soundData = torch.transpose(soundData, 0, 1).float()
    num_channels = 3

    specs = []
    for i in range(3): # we have 3 channels
        sr = 22050
        window_length = int(round(([25, 50, 100][i])*sr/1000))
        hop_length = int(round(([10, 25, 50][i])*sr/1000))

        spec = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=2205, win_length=window_length, hop_length=hop_length, n_mels=128)(soundData)
        eps = 1e-6
        spec = torch.log(spec + eps)
        spec = torchvision.transforms.Resize((128, 250))(spec)
        specs.append(torch.squeeze(spec))
    spec = torch.stack(specs)
    spec = normalize.forward(spec) 
    spec = spec.unsqueeze(0)
    
    threads.append(threading.Thread(target=run_model, args=(spec,threads)).start())

def record_audio(audio, threads):
    print("Recording audio...")
    sd.rec(samplerate=fs, out=audio)
    sd.wait()
    global stop_threads
    if stop_threads:
        return
    threads.append(threading.Thread(target=preprocesses_audio, args=(audio,threads)).start())
    threads.append(threading.Thread(target=record_audio, args=(audio,threads)).start())
    
threads = []

cb = lambda x : print(x)


t = threading.Thread(target=record_audio, args=(audio,threads))
threads.append(t)
t.start()

try:
    while 1:
        time.sleep(.1)
except KeyboardInterrupt:
    stop_threads = True