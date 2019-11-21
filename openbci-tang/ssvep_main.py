#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/14 16:33
# @Version : 1.0
# @File    : ssvep_with_openbci.py
# @Version : 1.0
# @License : (C) All Rights Reserved

from __future__ import division
from __future__ import print_function
from cyton import *
import numpy as np
from scipy import signal as scipy_signal
from ssvep_detector import SSVEPDETECT
from pynput.keyboard import Listener,Key
from eyeblink import DBlinkDetect
import serial


class KEYMONITOR():
    def __init__(self):
        self.esc_pressed = False
        # with Listener(on_press=self.press) as listener:
            # listener.join()
        self.listener = Listener(on_press=self.press)
        self.listener.start()

    def press(self,key):
        try:
            if key == Key.esc:
                self.esc_pressed = True
        except:
            pass
            
    def __del__(self):
        self.release()
    
    def release(self):
        self.listener.stop()

class LEDS():
    def __init__(self,com):
        self.serial = serial.Serial(com,9600)
        self.sr = ['\x00','\x01','\x02','\x03','\x04']
        self.tr = ['\x0a','\x0b','\x0c','\x0d','\x0e']

    def showr(self,r):
        if r < 5:
            self.serial.write(self.sr[r])

    def showt(self,t):
        if t < 5:
            self.serial.write(self.tr[t])

def main():
    eegchannles = [4,5,6,7]
    eogchannles = [0]

    led = LEDS('COM19')

    key_ = KEYMONITOR()
    ssvep_detector = SSVEPDETECT(srate=250,notch=True,freqs = [5.25,6.25,7.25,8.33],epoch=4)
    blink_detector = DBlinkDetect(srate=250,th_w = [60,250],th_h = 150)
    ob = OpenBCICyton(port='COM30')
    ob.start_record()

    c = 0
    cmd = 0
    while not key_.esc_pressed:
        c += 1
        raw = ob.read_data()         #行向量，每一行是一个通道数据
        eeg = raw[eegchannles,:]
        eog = raw[eogchannles,:]
        s_r = ssvep_detector.detect(eeg)
        e_r = blink_detector.detect_and_report(eog)

        c%=3
        if c==0:
            cmd = s_r
            led.showr(cmd)
        if e_r:
            led.showt(cmd)
    ob.stop()
    time.sleep(2)


def main2():
    k = KEYMONITOR()
    while not k.esc_pressed:
        pass
    # k.release()

if __name__ == '__main__':
    main()




