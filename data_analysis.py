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

def stft(sig,N_FFT,SAMPLE_RATE):
    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)
    return stft

def Mel_S(wav_file):

    # mel-spectrogram
    y, sr = librosa.load(wav_file, sr = 16000)
    wav_length = len(y) / sr
    #sr means sampling rate
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))
    S = librosa.feature.melspectrogram(y = y, n_mels = 128, n_fft = input_nfft, hop_length = input_stride)
    
    return S, wav_length

def MFCC(wav_file):

    # mel-spectrogram
    y, sr = librosa.load(wav_file, sr = 16000)
    wav_length = len(y) / sr
    #sr means sampling rate
    input_nfft = int(round(sr * frame_length))
    input_stride = len(y)//512 #int(round(sr * frame_stride))
    S = librosa.feature.mfcc(y =y, sr = sr, n_fft = input_nfft, n_mels = 128, hop_length = input_stride , n_mfcc = 13)
    return S, wav_length

path = "./sample_dataset/train/train_data/41_0508_171_0_08412_03.wav"

(rate, width, sig) = wavio.readwav(path)

sig = sig.ravel()
sig = sig.astype(np.float32)
#print(librosa.core.stft(sig, 512))
y, sr = librosa.load(path, sr = 16000)

#print(librosa.feature.mfcc(sig, sr = 16000))
x = get_spectrogram_feature(path)
#print(y, type(y), y.shape)
#print(sig,type(sig), sig.shape)
mel_x, w1 = Mel_S(path)
#mfcc_x = MFCC(path)

#print(x.shape, x.type)
print(x)
print(x.shape)
#print(mfcc_x.shape)

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
#ax1.pcolor(mfcc_x)
ax1.plot(sig)
ax2.pcolor(mel_x)
ax3.pcolor(x)
#plt.pcolor(x)
plt.show()
