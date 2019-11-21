#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/14 16:33
# @Version : 1.0
# @File    : ssvep_with_openbci.py
# @Version : 1.0
# @License : (C) All Rights Reserved

from cyton import *
from msvcrt import getch,kbhit
import numpy as np
from scipy import signal as scipy_signal
from cca import cca
from matplotlib import pyplot as plt


SRATE = 250
EPOCH = 4    #使用4秒的数据


# 模板
dataLength = EPOCH*SRATE
t = np.linspace(0,dataLength-1,dataLength)/SRATE
freqs = [5.25,6.25,7.25,8.33]
template = []
for f in freqs:
    tem = [np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t), np.sin(4*np.pi*f*t),np.cos(4*np.pi*f*t), np.sin(6*np.pi*f*t),np.cos(6*np.pi*f*t)]
    tem = np.vstack(tem)
    template.append(tem)

# notch filter
fs = SRATE/2.
Fo = 50
Q = 15
w0 = Fo / (fs)
notchB, notchA = scipy_signal.iirnotch(w0=w0, Q=Q)

# bandpass filter
Wp = np.array([3/fs, 90/fs])
Ws = np.array([2/fs, 100/fs])
N, Wn = scipy_signal.cheb1ord(Wp,Ws,3,40)
bpB,bpA = scipy_signal.cheby1(N,0.5,Wn,'bandpass')

# 绘图
# plt.ion()
# plt.figure(1)

def update(sig):
    # sig = sig.transpose()
    # plt.clf()
    # plt.plot(sig[:,1])
    # plt.pause(0.01)
    pass


    # m = np.mean(sig,axis=0)
    # sigg = sig - m
    # coef = np.ceil(100/(0.1 + np.max(np.abs(sigg),axis=0)))
    # siggg = coef * sigg
    #
    # plt.clf()
    # for i in range(8):
    #     plt.plot(siggg + 100 + i*200)
    # plt.pause(0.01)



class ShowCurve():
    def __init__(self):
        plt.ion()

    def update(self,sig):
        for i in range(8):
            plt.subplot(8,1,i)
            plt.clf()
            plt.plot(sig[i,:])
            plt.pause(0.001)


def process(signal):    #输入信号为行向量，每一行为一个通道数据
    sig = scipy_signal.filtfilt(notchB, notchA, signal)
    sig = scipy_signal.filtfilt(bpB, bpA, sig)
    sig = sig[:,-SRATE*EPOCH:]

    #update(sig)

    rr = []
    for i in range(4):
        A,B,r = cca(sig[-4:,:],template[i])
        rr.append(np.max(r))
    return np.argsort(rr)[-1]


def main():
    ob = OpenBCICyton(port='COM30')
    ob.start_record()

    eeg = np.zeros((1,8),np.float32)
    srate = 250

    while True:
        # esc = ord(getch())
        # if esc == 27: break

        raw = ob.read_data()

        tem = np.array(raw)
        eeg = np.vstack((eeg,tem))
        if eeg.shape[0] > srate * 5:
            eeg = eeg[-srate*5:,:]  #保留最后5秒数据
            res = process(eeg[:,].transpose())
            print res

    ob.stop()

def main1():
    while True:
        if kbhit():
            ch = ord(getch())
            # print ch
            if ch == '27':break
        time.sleep(0.5)
        print '?'

def main2():
    from pynput.keyboard import Listener
    stp = False
    def press(key):
        print 'yes'
        try:
            print key.char
        except:
            pass

    with Listener(on_press=press) as listener:
        listener.join()

    while not stp:
        time.sleep(0.5)




if __name__ == '__main__':
    main2()






