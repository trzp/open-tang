#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/16 10:56
# @Version : 1.0
# @File    : ssvep_detector.py
# @Author  : Jingsheng Tang
# @Version : 1.0
# @Contact : mrtang@nudt.edu.cn   mrtang_cs@163.com
# @License : (C) All Rights Reserved

import numpy as np
import scipy.signal as scipy_signal
from numpy import dot
import scipy.linalg as la


class SSVEPDETECT():
    def __init__(self,srate=250,notch=True,freqs = [5.25,6.25,7.25,8.33],epoch=4):
        self.srate = 250
        self.notch = notch

        # notch filter
        fs = self.srate / 2.
        Fo = 50
        Q = 15
        w0 = Fo / (fs)
        self.notchB, self.notchA = scipy_signal.iirnotch(w0=w0, Q=Q)

        # bandpass filter
        Wp = np.array([3. / fs, 90. / fs])
        Ws = np.array([2. / fs, 100. / fs])
        N, Wn = scipy_signal.cheb1ord(Wp, Ws, 3, 40)
        self.bpB, self.bpA = scipy_signal.cheby1(N, 0.5, Wn, 'bandpass')

        self.dataLength = epoch*self.srate
        self.headLength = self.srate
        self.processLength = self.dataLength + self.headLength

        t = np.linspace(0, self.dataLength - 1, self.dataLength) / self.srate
        self.freqs = freqs
        self.template = []
        for f in self.freqs:
            tem = [np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.sin(4 * np.pi * f * t),
                   np.cos(4 * np.pi * f * t), np.sin(6 * np.pi * f * t), np.cos(6 * np.pi * f * t)]
            tem = np.vstack(tem)
            self.template.append(tem)

        self.signal = []

    def detect(self,sig):    #输入信号为行向量，每一行为一个通道数据
        self.signal.append(sig)
        eeg = np.hstack(self.signal)
        if eeg.shape[1] < self.processLength:
            self.signal = [eeg]
            return -1
        self.signal = [eeg[:, -self.processLength:]]
        if self.notch:
            sig = scipy_signal.filtfilt(self.notchB, self.notchA, eeg)
        sig = scipy_signal.filtfilt(self.bpB, self.bpA, sig)
        sig = sig[:, -self.dataLength:]

        rr = []
        for temp in self.template:
            A, B, r = cca(sig,temp)
            rr.append(np.max(r))
        return np.argsort(rr)[-1]


def clean_and_sort_eigenvalues(eigenvalues, eigenvectors):
    evs = [(va, ve) for va, ve in zip(eigenvalues, eigenvectors.T) if va.imag == 0]
    evs.sort(key=lambda evv: evv[0], reverse=True)
    sevals = np.array([va.real for va, _ in evs])
    sevecs = np.array([ve for _, ve in evs]).T
    return sevals, sevecs


def cca(X, Y):
    """Canonical Correlation Analysis

    :param X: observation matrix in X space, every column is one data point
    :param Y: observation matrix in Y space, every column is one data point

    :returns: (basis in X space, basis in Y space, correlation)
    """

    N = X.shape[1]
    Sxx = 1.0 / N * dot(X, X.T)
    Sxy = 1.0 / N * dot(X, Y.T)
    Syy = 1.0 / N * dot(Y, Y.T)

    epsilon = 1e-6
    rSyy = Syy + epsilon * np.eye(Syy.shape[0])
    rSxx = Sxx + epsilon * np.eye(Sxx.shape[0])
    irSyy = la.inv(rSyy)

    L = dot(Sxy, dot(irSyy, Sxy.T))
    lambda2s, A = la.eig(L, rSxx)
    lambdas = np.sqrt(lambda2s)
    clambdas, cA = clean_and_sort_eigenvalues(lambdas, A)
    B = dot(irSyy, dot(Sxy.T, dot(cA, np.diag(1.0 / (clambdas + 1e-8)))))

    return (cA, B, clambdas)


