from loader import get_script, get_spectrogram_feature
import label_loader
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wavio
import torch
import numpy as np

frame_length = 0.025
frame_stride = 0.010
N_FFT = 512
SAMPLE_RATE = 16000

def stft(filepath):
    
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()

    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)
    print(stft.shape)
    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5);
    amag = stft.numpy();
    feat = torch.FloatTensor(amag)
    #feat = torch.FloatTensor(feat).transpose(0, 1)

    return feat

def Mel_S(filepath):

    # mel-spectrogram
    sig, sr = librosa.load(filepath, sr = 16000)
    #(rate, width, sig) = wavio.readwav(filepath)
    #sig = sig.ravel()
    #wav_length = len(y) / sr
    #sr means sampling rate
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))
    S = librosa.feature.melspectrogram(y = y, n_mels = 40, n_fft = input_nfft, hop_length = input_stride)
    S_po = librosa.power_to_db(S, ref = np.max)
    return S, S_po

def MFCC(wav_file):

    # mel-spectrogram
    (rate, width, sig) = wavio.readwav(wav_file)
    sig = sig.ravel()

    #y, sr = librosa.load(wav_file, sr = 16000)
    wav_length = len(y) / sr
    #sr means sampling rate
    input_nfft = int(round(sr * frame_length))
    input_stride = len(y)//512 #int(round(sr * frame_stride))
    S = librosa.feature.mfcc(y =y, sr = sr, n_fft = input_nfft, n_mels = 40, hop_length = input_stride , n_mfcc = 13)
    return S

path = "./sample_dataset/train/train_data/wav_001.wav"

(rate, width, sig) = wavio.readwav(path)

sig = sig.ravel()
sig = sig.astype(np.float32)
#print(librosa.core.stft(sig, 512))
y, sr = librosa.load(path, sr = 16000)

#print(librosa.feature.mfcc(sig, sr = 16000))
x = stft(path)
#print(y, type(y), y.shape)
#print(sig,type(sig), sig.shape)
mel_x, db = Mel_S(path)
mfcc_x = MFCC(path)

#print(x.shape, x.type)
print(db)
print(mel_x.shape)
print(mfcc_x.shape)
#print(mfcc_x.shape)

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
#ax1.pcolor(mfcc_x)
ax1.pcolor(mfcc_x)
ax2.pcolor(mel_x)
ax3.pcolor(x)
#plt.pcolor(x)
plt.show()
