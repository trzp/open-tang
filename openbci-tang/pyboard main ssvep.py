#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/11 15:22
# @Version : 1.0
# @File    : main.py
# @Author  : Jingsheng Tang
# @Version : 1.0
# @Contact : mrtang@nudt.edu.cn   mrtang_cs@163.com
# @License : (C) All Rights Reserved

import pyb
from machine import UART
import time

class Flicker():
    def __init__(self,p,freq,time_scale,alive_mode = 2,*args):
        # 用于向引脚p输出freq频率的电平翻转（初始高电平）等
        # time_scale: 定时器的定时间隔
        # 支持三种模式alive_mode：0：常亮常灭模式
        #                         1：有限生命模式，按照设定的频率闪烁一定时间args参数指定
        #                         2: 永生模式
        # args: 有限生命模式下，通过参数指定生命周期,如果不指定默认为1秒，也可在set_mode中随时指定

        self.pin = pyb.Pin(p,pyb.Pin.OUT_PP)
        self.pin.high()
        self.count = int(500./(freq*time_scale))
        self.clk = 0

        self.ALIVE_COUNT = 0
        self.alive_count = 0

        self.alive_mode = alive_mode
        if alive_mode == 1:
            if len(args)>0:
                alive_count = args[0]
            else:
                alive_count = int(1000/time_scale)

            self.ALIVE_COUNT = int(alive_count/time_scale)
            self.alive_count = int(alive_count/time_scale)


    def toggle(self):
        self.pin.low() if self.pin.value() else self.pin.high()

    def __once(self):
        self.clk += 1
        if self.clk >= self.count:
            self.clk = 0
            self.toggle()

    def set_mode(self,mode,*args):  # mode: 0 常亮常灭模式，1 有限生命模式，2 永生模式
        self.alive_mode = mode
        if mode == 0:   #常亮常灭
            on = args[0]
            self.pin.low() if on else self.pin.high()
        elif mode == 1: #有限生命
            self.reset()
            if len(args)>0:     #生命周期由参数指定
                self.alive_count = args[0]
            else:               #生命周期由初始化参数指定
                self.alive_count = self.ALIVE_COUNT
        elif mode == 2:
            self.reset()
        else:
            pass

    def update(self):
        if self.alive_mode == 0:
            return
        elif self.alive_mode == 1:
            self.alive_count -= 1
            if self.alive_count > 0:    #生命期内
                self.__once()
            elif self.alive_count == 0: #生命终止
                self.reset()            #状态初始化
            else:                       #生命终止继续
                self.alive_count = 0   #避免重复reset
        elif self.alive_mode == 2:
            self.__once()   # 永生模式
        else:
            pass

    def reset(self):
        self.pin.low()
        self.clk = 0


hzlist = [2,2,2,2,2,5.25,6.25,7.25,8.25,12.25]
pinlist = ['X1','X2','X3','X4','X5','X6','X7','X8','Y9','Y10']
mode = [0,0,0,0,0,2,2,2,2,2]

timer_freq = 2000
stp = 1000/timer_freq
flickers = []

for i,hz in enumerate(hzlist):
    flickers.append(Flicker(pinlist[i],hz,stp,mode[i]))

def t_callback(timer):
    for flicker in flickers:
        flicker.update()

tim = pyb.Timer(1,freq = timer_freq)   # 2000Hz对应定时间隔0.5ms
tim.callback(t_callback)

# 上位机通过串口发送ascii码来指示pyboard显示当前ssvep的分类结果和当前执行的任务
# '0','1','2'.... 对应实际的0,1,2...LED, 指示其常亮
# '10','11',...   对应指示相关的LED闪烁表示执行该命令，指示其闪烁
uart = UART(1, 9600)
a = 0
while True:
    buf = uart.read()
    if buf is not None:
        cmd = int(buf[-1])
        for i in range(5):
            flickers[i].set_mode(0,False)   #全部复位
        if cmd < 5:
            flickers[cmd].set_mode(0,True)
        else:
            flickers[cmd-10].set_mode(1,1500)
    time.sleep_ms(50)
    a += 1
    print(a)





