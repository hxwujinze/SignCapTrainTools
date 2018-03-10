# coding:utf-8
from __future__ import division
from __future__ import print_function
from myo import init, Hub, Feed, StreamEmg
import time
import pythoncom
import pyHook
import threading
import os
import shutil

tagGesture = 0
tagNext = 0
capture_frequency = 100

class Myos(object):
    def __init__(self, myo_device):
        self._myo_device = myo_device
        self._time = 0
        self._t_s = 1 / capture_frequency
        self.Emg = []
        self.Acc = []
        self.Gyr = []
        self.output = list()

    def start(self):
        startTime = time.time()
        printTime = time.time()
        printlist = ["|", "/", "-", "\\", "-> "]
        i = 0
        while True:
            currentTime = time.time()
            if (currentTime - startTime) > self._t_s:
                startTime = time.time()
                self.writeAllData()
            if currentTime - printTime > 0.5 and tagGesture == 1:
                printTime = time.time()
                print('\b', end='')
                print(printlist[i], end='')
                i = i + 1
                if i > 4:
                    i = 0
                    print(printlist[i], end='')

    def getAllData(self):
        global tagNext
        self.Emg = self._myo_device[0].emg + (tagNext,)
        self.Acc = [it for it in self._myo_device[0].acceleration]
        self.Gyr = [it for it in self._myo_device[0].gyroscope]

    def writeAllData(self):
        global tagGesture
        global tagNext
        if tagGesture == -1:
            savefile(self)
            tagGesture = 0
        else:
            self.getAllData()
            self.output.append(self.Emg)
            self.output.append(self.Acc)
            self.output.append(self.Gyr)

def savefile(capture_obj):
    for index, r in enumerate(["Emg#.txt", "Acceleration#.txt", "Gyroscope#.txt"]):
        with open(r, "a") as text_file:
            text_file.write("{0}\n".format(capture_obj.output[index]))
    path = os.getcwd()
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
    global tagGesture
    global tagNext
    # print "Key:", event.Key

    if event.Key == "Escape":
        if tagGesture != -1:
            print("\n###  gesture end and save files:  ###")
        tagGesture = -1

    if event.Key == "Return":
        if tagGesture != 1:
            print(" ")
            print("#########  gesture start:  ##########")
            tagNext = 0
        tagGesture = 1

    if event.Key == "T":
        if tagGesture == 1:
            print("\bTag " + str(tagNext + 1) + " ")
            tagNext += 1

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
