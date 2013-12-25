#!/usr/bin/python
# coding=UTF-8

import sys
import numpy as np
import scipy as sp
import pylab
from itertools import izip
import wave
from sklearn import svm
import random
import time

TYPES_NUM = 6

def file_type(line):
    t = unicode(line.split('1')[0])
    # print 't=%s'%t
    return {u'a':0,u'i':1,u'o':2,u'u':3,u'ih':4,u'eh':5}[t]

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

def predict_quality(data,test):
    l = min(len(data),len(test))
    correct = 0
    for i in range(l):
        if data[i] == test[i]: correct+=1
    return 100*correct/float(l)


def main():
    input = []
    correct_types = []
    strict_input = []
    strict_types = []
    FILES = 'files.dat'

    random.seed(time.time())

    # file_list = open(FILES,'r')
    file_list = sys.argv[1:]
    for line in file_list:
        f_name = line
        f_type = file_type(line)
        # sys.stderr.write("%s %d\n"%(f_name,f_type))

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
        svm_data = np.array(sm_amp)

        input.append(svm_data)
        correct_types.append(f_type)
        if random.random() >= 0.5:
            strict_input.append(svm_data)
            strict_types.append(f_type)

        data_file.close()

    if len(input) == 0:
        sys.stderr.write("No input files\n")
        exit(1)

    svc = svm.SVC(kernel='linear')    # linear OR poly OR rbf
    # svc = svm.SVC(kernel='poly',degree=2) 
    # svc = svm.SVC(kernel='rbf')

    # svc.fit(input, types)
    svc.fit(strict_input,strict_types)

    new_types=svc.predict(input)

    for i in range(TYPES_NUM):
        print new_types[i*10:i*10+10]

    sys.stderr.write("%2.2f\n"%predict_quality(new_types,correct_types))




    
    # for r in A:
    #     t = types.pop(0)
    #     print "%.06f\t%.06f\t%d" % (r[0], r[1], t)

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