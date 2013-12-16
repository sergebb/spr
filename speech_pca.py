#!/usr/bin/python

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

def main():
    input = None
    types = list()
    FILES = 'files.dat'

    file_list = open(FILES,'r')
    for line in file_list:
        f_name = line.split(' ')[0]
        f_type = int(line.split(' ')[1])

        types.append(f_type)

        data_file = wave.open(f_name,'r')

        frame_rate = data_file.getframerate()
        frames = data_file.readframes(data_file.getnframes())
        signal = np.fromstring(frames, 'Int16')

        Pxx, freqs, bins, im=pylab.specgram(signal, Fs=frame_rate)
        combined_amp = np.amax(Pxx,axis=1)

        if input == None:
            input = combined_amp
        else:
            input = np.vstack([input, combined_amp])

        data_file.close()

    A = pca(input)

    for r in A:
        t = types.pop(0)
        print "%.04f\t%.04f\t%d" % (r[0], r[1], t)

    file_list.close()

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