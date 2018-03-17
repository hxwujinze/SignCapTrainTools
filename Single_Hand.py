# coding:utf-8
from __future__ import division
from __future__ import print_function

import os
import shutil
import threading
import time

import pyHook
import pythoncom
from myo import init, Hub, Feed, StreamEmg

DATA_DIR_PATH = os.getcwd()

CAPTURE_FREQUENCY = 100

STATE_END_OF_CAPTURE = -1
STATE_START_CAPTURE = 0
STATE_CAPTURING = 1

capture_state = 0
next_tag_num = 0

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

    def writeAllData(self):
        global capture_state
        global next_tag_num
        if capture_state == STATE_END_OF_CAPTURE:
            savefile(self)
            capture_state = STATE_START_CAPTURE
            self.Emg = []
            self.Acc = []
            self.Gyr = []
        else:

            if capture_state == STATE_CAPTURING:
                self.getAllData()

    def get_capture_data_list(self):
        return [self.Emg, self.Acc, self.Gyr]

def savefile(capture_obj):
    for index, r in enumerate(["Emg#.txt", "Acceleration#.txt", "Gyroscope#.txt"]):
        with open(r, "w") as text_file:
            # [ emg, acc, gyr]
            data_out_put = capture_obj.get_capture_data_list()
            str_res = ''
            for each_line in data_out_put[index]:
                str_res += (str(each_line) + '\n')
            text_file.write(str_res)
    path = DATA_DIR_PATH
    OriFiles = os.listdir(path)
    fileList = [file for file in OriFiles if file.endswith("#.txt")]
    for file in fileList:
        sourceFile = os.path.join(path, file)
        targetFile = os.path.join(path + "\\" + file[:-5], file)
        if os.path.isfile(sourceFile):
            if not os.path.exists(path + "\\" + file[:-5]):
                os.makedirs(path + "\\" + file[:-5])
            shutil.move(sourceFile, targetFile)
            os.rename(targetFile, os.path.join(path + "\\" + file[:-5], "0.txt"))
            tarFiles = os.listdir(path + "\\" + file[:-5])
            tarFiles.sort(key=lambda x: int(x[:-4]))
            tarSort = tarFiles
            print(tarSort)
            last = int(tarSort[-1][:-4])
            if last == 0:
                os.rename(os.path.join(path + "\\" + file[:-5], "0.txt"),
                          os.path.join(path + "\\" + file[:-5], "1.txt"))
            else:
                os.rename(os.path.join(path + "\\" + file[:-5], "0.txt"),
                          os.path.join(path + "\\" + file[:-5], str(last + 1) + ".txt"))

def getkeyboard():
    hm = pyHook.HookManager()
    hm.KeyDown = onKeyboardEvent
    hm.HookKeyboard()
    pythoncom.PumpMessages()

def onKeyboardEvent(event):
    global capture_state
    global next_tag_num
    # print "Key:", event.Key

    if event.Key == "Escape":
        if capture_state != STATE_END_OF_CAPTURE:
            print("\n###  gesture end and save files:  ###")
            capture_state = STATE_END_OF_CAPTURE

    if event.Key == "Return":
        if capture_state != STATE_CAPTURING:
            print(" ")
            print("#########  gesture start:  ##########")
            next_tag_num = 0
            capture_state = STATE_CAPTURING

    if event.Key == "T":
        if capture_state == STATE_CAPTURING:
            print("\bTag " + str(next_tag_num + 1) + " ")
            next_tag_num += 1

if __name__ == "__main__":
    # 使用debug模式可以正常工作？？？
    init(os.path.dirname(__file__))
    feed = Feed()
    hub = Hub()
    times = 0
    hub.run(1000, feed)
    try:
        myo_device = feed.get_devices()
        print(myo_device)
        time.sleep(1)
        myo_device[0].set_stream_emg(StreamEmg.enabled)
        t1 = threading.Thread(target=getkeyboard)
        t1.setDaemon(True)
        t1.start()
        time.sleep(0.5)
        twoMyos = Myos(myo_device)
        twoMyos.start()
    except KeyboardInterrupt:
        print("Quitting ...")
    finally:
        hub.shutdown()

