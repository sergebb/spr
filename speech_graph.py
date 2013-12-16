#!/usr/bin/python

import sys
import numpy as np
import scipy as sp
from itertools import izip
import wave

import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def Extremes(data,diff):
        minpoints = []
        maxpoints = []
        lastmax = lastmin = 1
        lookMax = True
        lastIsMax = True
        for i in range(1,len(data)-1):
            if data[i] > data[i-1] and data[i] > data[i+1] and lookMax == True:
                lookMax = False

                if data[i] - lastmin > diff:
                    if len(maxpoints)>0 and lastIsMax == True and lastmax < data[i]:
                        maxpoints.pop()
                        maxpoints.append(i)
                        lastmax = data[i]
                        lastIsMax = True
                    elif len(maxpoints)==0 or lastIsMax == False:
                        maxpoints.append(i)
                        lastmax = data[i]
                        lastIsMax = True
            if data[i] < data[i-1] and data[i] < data[i+1] and lookMax == False:
                lookMax = True

                if lastmax - data[i] > diff:
                    if len(minpoints)>0 and lastIsMax == False and lastmin > data[i]:
                        minpoints.pop()
                        minpoints.append(i)
                        lastmin = data[i]
                        lastIsMax = False
                    elif len(minpoints)==0 or lastIsMax == True:
                        minpoints.append(i)
                        lastmin = data[i]
                        lastIsMax = False

        return maxpoints, minpoints

def Formants(data):
    limits=[]
    formants=[]
    limits.append(0)
    maxp,minp = Extremes(data,50)
    for i in minp:
        if data[i]<10 and i-limits[-1]>10:
            limits.append(i)
    if len(data)-limits[-1]>10:
        limits.append(len(data)-1)

    print limits

    for (p,q) in izip(limits,limits[1:]):
        formants.append( p+np.argmax(data[p:q]) )

    return formants


def main():
    try:
        data_file = sys.argv[1]
        audio = wave.open(data_file,'r')
    except:
        sys.stderr.write('Wrong file name, /script.py file \n')
        return 1

    for i in range(1,len(sys.argv)):
        print sys.argv[i]
        audio = wave.open(sys.argv[i],'r')

        file_num = sys.argv[i].split('.')[0]

        img_name = file_num + '.png'

        # print "Channels = %d"%audio.getnchannels()
        # print "Sample width = %d"%audio.getsampwidth()
        # print "Sampling frequency = %d"%audio.getframerate()
        # print "Number of audio frames = %d"%audio.getnframes()
        # print "Compression type = %s"%audio.getcompname()

        frame_rate = audio.getframerate()

        frames = audio.readframes(audio.getnframes())
        signal = np.fromstring(frames, 'Int16')
        time=np.linspace(0, len(signal)/frame_rate, num=len(signal))

        Pxx, freqs, bins, im=pylab.specgram(signal, Fs=frame_rate)

        combined_amp = np.amax(Pxx,axis=1)
        deg = 3
        sm_amp = smoothListGaussian(combined_amp,degree=deg)

        print freqs[Formants(combined_amp)]
        plt.plot(freqs,combined_amp)
        # # plt.plot(freqs,Pxx)
        # for i in range(len(combined_amp)):
        #     print "%04d\t%04d" % (i,combined_amp[i])


        # plt.plot(time,signal)
        # pylab.plot(frames,len(frames))
        # fft = np.real(np.fft.fft(signal))
        # freq = np.fft.fftfreq(len(signal))
        # freq_in_hertz=np.abs(freq*frame_rate)
        # plt.plot(freq_in_hertz,fft)


        # print freqs[np.argmax(combined_amp)]

        plt.show()
        # plt.savefig(img_name)
        # plt.clf()
        audio.close()
    
    pass


if __name__ == "__main__":
    main()