#!/usr/bin/python
# coding=UTF-8

import sys
import numpy as np
import scipy as sp
import pylab
from itertools import izip
import wave

def pca(input):
    input = input - input.mean(axis=0)
    (u,s,v) = np.linalg.svd(input, full_matrices=False)
    return np.dot(input, v.T)

def Normalize(data):
    num = data.shape[0]
    norm = np.amax(data)/100.0

    for i in range(num):
        data[i]/=norm
    return data

def file_type(line):
    t = unicode(line.split('1')[0])
    # print 't=%s'%t
    return {u'a':1,u'i':2,u'o':3,u'u':4,u'ih':5,u'eh':6}[t]

def smoothListGaussian(list,strippedXs=False,degree=5):  
     window=degree*2-1  
     weight=np.array([1.0]*window)  
     weightGauss=[]  
     for i in range(window):
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(np.exp((4*(frac))**2))  
         weightGauss.append(gauss)  
     weight=np.array(weightGauss)*weight  
     smoothed=[0.0]*(len(list)-window)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)  
     return smoothed

def main():
    input = None
    types = list()
    FILES = 'files.dat'

    # file_list = open(FILES,'r')
    file_list = sys.argv[1:]
    for line in file_list:
        f_name = line
        f_type = file_type(line)
        sys.stderr.write("%s %d\n"%(f_name,f_type))

        types.append(f_type)

        data_file = wave.open(f_name,'r')

        frame_rate = data_file.getframerate()
        frames = data_file.readframes(data_file.getnframes())
        signal = np.fromstring(frames, 'Int16')

        Pxx, freqs, bins, im=pylab.specgram(signal, Fs=frame_rate)
        combined_max_amp = np.amax(Pxx,axis=1)
        combined_mean_amp = np.mean(Pxx,axis=1)

        deg = 10
        sm_amp = smoothListGaussian(combined_mean_amp,degree=deg)

        # pca_data = Normalize(sm_amp)
        pca_data = Normalize( np.array(sm_amp))

        if input == None:
            input = pca_data
        else:
            input = np.vstack([input, pca_data])

        data_file.close()

    if input == None:
        sys.stderr.write("No input files\n")
        exit(1)

    A = pca(input)

    for r in A:
        t = types.pop(0)
        print "%.06f\t%.06f\t%d" % (r[0], r[1], t)

    # file_list.close()

    # try:
    #     data_file = sys.argv[1]
    #     audio = wave.open(data_file,'r')
    # except:
    #     sys.stderr.write('Wrong file name, /script.py file \n')
    #     return 1

    # print "Channels = %d"%audio.getnchannels()
    # print "Sample width = %d"%audio.getsampwidth()
    # print "Sampling frequency = %d"%audio.getframerate()
    # print "Number of audio frames = %d"%audio.getnframes()
    # print "Compression type = %s"%audio.getcompname()

    # frame_rate = audio.getframerate()

    # frames = audio.readframes(audio.getnframes())
    # signal = np.fromstring(frames, 'Int16')
    # time=np.linspace(0, len(signal)/frame_rate, num=len(signal))

    # Pxx, freqs, bins, im=pylab.specgram(signal, Fs=frame_rate)

    # combined_amp = np.amax(Pxx,axis=1)
    # plt.plot(freqs,combined_amp)
    # # plt.plot(time,signal)
    # # pylab.plot(frames,len(frames))
    # # fft = np.real(np.fft.fft(signal))
    # # freq = np.fft.fftfreq(len(signal))
    # # freq_in_hertz=np.abs(freq*frame_rate)
    # # plt.plot(freq_in_hertz,fft)


    # print freqs[np.argmax(combined_amp)]

    # plt.show()
    # audio.close()
    
    # pass


if __name__ == "__main__":
    main()