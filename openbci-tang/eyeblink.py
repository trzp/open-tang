#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/15 22:24
# @Version : 1.0
# @File    : eyeblink.py
# @Author  : Jingsheng Tang
# @Version : 1.0
# @Contact : mrtang@nudt.edu.cn   mrtang_cs@163.com
# @License : (C) All Rights Reserved

from __future__ import division
from __future__ import print_function
from scipy import signal as scipy_signal
import numpy as np
import time

from matplotlib import pyplot as plt
# plt.ion()
# plt.figure()

# 基本原理
# 1-5hz滤波，实测可以。理论上眨眼频率可达4Hz左右
# 降采样
# 峰值检测（波峰波谷）
# 对波峰-波谷（可表征一次眨眼）的幅值差和宽度进行判断，没有使用绝对的幅值，可靠性更高

class EyeBlink():
    def __init__(self,srate=250,filterdata = True,th_w = [60,250],th_h = 150):
        self.srate = 250
        self.filterdata = filterdata

        # self.ttemp = []

        # notch filter
        fs = self.srate / 2.
        Fo = 50
        Q = 15
        w0 = Fo / (fs)
        self.notchB, self.notchA = scipy_signal.iirnotch(w0=w0, Q=Q)

        # bandpass filter
        Wp = np.array([1 / fs, 5 / fs])
        Ws = np.array([0.5 / fs, 10 / fs])
        N, Wn = scipy_signal.cheb1ord(Wp, Ws, 3, 40)
        self.bpB, self.bpA = scipy_signal.cheby1(N, 0.5, Wn, 'bandpass')

        # 均值降采样
        targetSrate = 50
        self.meanNum = int(self.srate / targetSrate) # 将信号降采样到50Hz附近
        self.meanNumf = np.ones(self.meanNum)/self.meanNum

        tu = self.meanNum * 1000 / self.srate # 降采样后采样点间隔时间
        self.th_w = [int(i/tu) for i in th_w]
        self.th_h = th_h

    def filter_eeg(self,sig):
        sig = scipy_signal.filtfilt(self.notchB, self.notchA, sig)
        sig = scipy_signal.filtfilt(self.bpB, self.bpA, sig)
        return sig

    def detect(self,sig):
        if self.filterdata:
            sig = self.filter_eeg(sig)
        sig = scipy_signal.lfilter(self.meanNumf,1,sig)
        sig = sig[self.meanNum-1::self.meanNum]    #均值降采样
        locs = scipy_signal.find_peaks(sig)[0]
        pks = sig[locs]
        locs1 = scipy_signal.find_peaks(-1*sig)[0]
        pks1 = sig[locs1]
        tem = np.vstack([locs,pks,np.ones(locs.size)])
        tem1 = np.vstack([locs1,pks1,-1*np.ones(locs1.size)])
        ppks = np.hstack([tem,tem1])
        ppks = ppks[:,np.argsort(ppks[0,:])]
        if ppks[2,0]==-1:
            ppks = ppks[:,1:]
        dppks = np.diff(ppks,axis=-1)
        ind = np.where((dppks[0,:]>self.th_w[0]) & (dppks[0,:]<self.th_w[1]) & (dppks[1,:]<-self.th_h))[0]
        if ind.size>0:
            blink = ppks[:2,ind]
            blink[0,:] = blink[0,:] * self.meanNum + self.meanNum - 1

            # self.ttemp.append(dppks[1,ind])
            return blink
        else:
            return None


class DBlinkDetect():
    #2次连续眨眼检测
    #上一次检测为True后，检测到一个False后，下一个True是一个新的双眨眼
    #换句话说False,True组合为一个新的双眨眼（上升沿）

    def __init__(self, srate=250, th_w=[60, 250], th_h=150):
        self.srate = srate
        self.eye_detector = EyeBlink(srate, False, th_w, th_h)
        self.signal = []

        self.dataLength = int(self.srate)  # 默认对1秒内的数据进行分析
        self.processLength = self.srate * 4

        self._N = 2*400/(1000/self.srate)   # 400ms为设定的一次眨眼最长时间

        self.cmd = [False]

    def detect(self, sig):  # 行向量，单通道, 返回原始判断
        sig = sig.flatten()
        self.signal.append(sig)
        eeg = np.hstack(self.signal)  # 将信号拼接到尾部
        if eeg.shape[-1] < self.processLength:  # 开始处理信号的条件
            self.signal = [eeg]
            return False
        self.signal = [eeg[-self.processLength:]]

        eeg = self.eye_detector.filter_eeg(eeg)  # 滤波处理
        eeg = eeg[-self.dataLength:]  # 头部信号去除，避免滤波畸变的影响

        blink = self.eye_detector.detect(eeg)  # 检测眨眼信号
        if blink is not None:
            ind = np.where(np.diff(blink[0,:])<self._N)[0]
            if ind.size > 0:
                return True    #检测到双眨眼
            else:
                return False
        else:
            return False

    def detect_and_report(self,sig):    #返回上升沿的检查结果，不会重复报告
        self.cmd.append(self.detect(sig))
        self.cmd = self.cmd[-2:]
        if not self.cmd[0] and self.cmd[1]: #[False,True]
            return True
        else:
            return False


class EyeBlinkDetect():
    # 将所有检测到的眨眼映射到全局时间轴上，但分段检测信号拼接起来带来混叠以及阈值改变问题没有完全解决

    # 对连续信号进行分析时，信号包可能将一次有效的眨眼信号分割开
    def __init__(self,srate=250,th_w = [60,250],th_h = 150):
        self.srate = srate
        self.eye_detector = EyeBlink(srate,False,th_w,th_h)
        self.signal = []

        self.dataLength = int(self.srate*0.6)   # 默认对0.6秒内的数据进行分析
        self.processLength = self.srate * 2
        self.blinkResultLen = int(1.5*self.srate)   # 总是将检测的结果映射到最近的1.5秒时间轴上
        self.BLINKDATA = []

        self.recordN = 0    # 记录处理的连续信号的长度

        self.__N = int(100/(1000./self.srate))   # 100ms不可能同时眨眼两次，用来消除混叠检测时的错位

    def detect(self,sig):   #行向量，单通道,注意默认分析最新的0.6秒数据，因此必须确保每次调用间隔低于0.6秒，否则会造成中间丢失掉某些信号未分析到
        inN = sig.size
        self.recordN += inN  # 记录总信号的长度，提供全局时间轴

        self.signal.append(sig)
        eeg = np.hstack(self.signal)            #将信号拼接到尾部
        if eeg.shape[0] < self.processLength:   #开始处理信号的条件
            self.signal = [eeg]
            return None
        self.signal = [eeg[-self.processLength:]]

        eeg = self.eye_detector.filter_eeg(eeg) #滤波处理
        eeg = eeg[-self.dataLength:]            #头部信号去除，避免滤波畸变的影响
        blink = self.eye_detector.detect(eeg)   #检测眨眼信号
        if blink is not None:
            startp = self.recordN - self.dataLength  # 该点和本次信号处理的0点式对齐的
            blink[0,:] += startp                #将当前检测到的结果映射到全局时间轴上
            self.BLINKDATA.append(blink)
            kk = np.hstack(self.BLINKDATA)
            # kk = self.__calibrate(kk)
            self.BLINKDATA = [kk]

    def getall(self):
        return self.__calibrate(self.BLINKDATA[0])

    def __calibrate(self,blink):
        blink = blink[:,np.argsort(blink[0,:])] #按第0行排序
        ind = np.where(np.diff(blink[0,:])>self.__N)[0]
        return blink[:,ind]


def main1():
    eog = np.load('eog.npy')
    signal = eog[:, 0].transpose()
    eye = EyeBlink(srate=250, filterdata=True)
    eye.detect(signal)

def main2():
    eye = EyeBlinkDetect(srate=250, th_w=[60, 250], th_h=120)
    eog = np.load('eog.npy')
    signal = eog[:, 0].transpose()
    N = signal.shape[-1]
    for i in range(0, N - 75, 75):
        sig = signal[i:i + 75]
        eye.detect(sig)

    bb = eye.getall()

def main3():
    eye = DBlinkDetect()
    # eog = np.load('eog.npy')
    eog = np.load('test.npy')
    signal = eog[:, 0].transpose()
    N = signal.shape[-1]

    ll = []
    for i in range(0, N - 75, 75):
        sig = signal[i:i + 75]
        res = eye.detect_and_report(sig)
        if res:
            print(res)
        ll.append(res)

    pass


if __name__ == '__main__':
    main3()





