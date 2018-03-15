# coding:utf-8
import time
import tkinter
from tkinter.constants import *

CAPTURE_FREQUENCY = 100

STATE_END_OF_CAPTURE = -1
STATE_START_CAPTURE = 0
STATE_CAPTURING = 1

class CaptureObj:
    def __init__(self):
        self.capture_batch = get_max_batch_num()
        self.capture_data = {}
        self.curr_capture_sign_num = 1

    def save_to_file(self):
        pass

class Myos(object):

    def __init__(self, myo_device):
        self._myo_device = myo_device
        self._time = 0
        self._t_s = 1 / CAPTURE_FREQUENCY
        self.Emg = []
        self.Acc = []
        self.Gyr = []

    def start(self):
        startTime = time.time()
        printTime = time.time()

        while True:
            currentTime = time.time()
            if (currentTime - startTime) > self._t_s:
                startTime = time.time()
                self.writeAllData()
            if currentTime - printTime > 0.5 and capture_state == 1:
                printTime = time.time()
                print(' . ')

    def getAllData(self):
        global next_tag_num
        self.Emg.append(self._myo_device[0].emg + (next_tag_num,))
        self.Acc.append([it for it in self._myo_device[0].acceleration])
        self.Gyr.append([it for it in self._myo_device[0].gyroscope])

def get_max_batch_num():
    file = open('batch_num', 'r')
    num = int(file.readline())
    file.close()
    num += 1
    file = open('batch_num', 'w')
    file.write(str(num) + '\n')
    file.close()
    return num

def start_cap():
    pass

if __name__ == '__main__':
    window = tkinter.Tk()

    window.title('collect')
    window.geometry('480x600')
    frame = tkinter.Frame(window, relief=RIDGE, borderwidth=2)
    frame.pack(fill=BOTH, expand=1)

    label = tkinter.Label(frame, text="Hello, World")
    label.pack(fill=X, expand=1)

    button_frame = tkinter.Frame(window)
    button_frame.pack()

    button_start_capture = tkinter.Button(button_frame, text='开始采集')
    button_start_capture.pack(side=LEFT)

    button_next_capture = tkinter.Button(button_frame, text='下一个手势')
    button_next_capture.pack(side=LEFT)

    button_save = tkinter.Button(button_frame, text='保存文件')
    button_save.pack(side=LEFT)

    button_new_cap = tkinter.Button(button_frame, text='新建采集')
    button_new_cap.pack(side=LEFT)

    button = tkinter.Button(button_frame, text="退出", command=window.destroy)
    button.pack(side=LEFT)

    window.mainloop()
