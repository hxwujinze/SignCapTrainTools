# coding:utf-8
import Tkinter
import os
import pickle
import threading
import time
from Tkinter import TOP, LEFT, END

import myo
from myo import VibrationType

DATA_PATH = os.getcwd() + '\\data'
TYPE_LIST = ['acc', 'emg', 'gyr']
GESTURES_TABLE = ['肉 ', '鸡蛋 ', '喜欢 ', '您好 ', '你 ', '什么 ', '想 ', '我 ', '很 ', '吃 ',
                  '老师 ', '发烧 ', '谢谢 ', '空手语']
GESTURES_SELECTED_LIST = [0 for i in range(len(GESTURES_TABLE))]


SIGN_COUNT = 14
CAPTURE_TIMES = 3

STATE_END_OF_CAPTURE = -1
# 一个手势的一次采集结束
STATE_STANDBY = 57
# 等待进行删除或者 保存 进行下一次采集
# 下一次采集可能是这个这个手语的下一次采集 也可能是下一个手语的第一次采集
STATE_START_CAPTURE = 0
# 一次采集的开始 进入CAPTURING循环
STATE_CAPTURING = 1
# 以100hz频率 在采集中的状态
STATE_STOP_COLLECTION = 51
# 结束整个采集过程


STATE_TABLE = {
    STATE_END_OF_CAPTURE: '数据采集结束',
    STATE_STOP_COLLECTION: '终止采集',
    STATE_CAPTURING: '数据采集中',
    STATE_START_CAPTURE: '数据采集开始',
    STATE_STANDBY: '数据采集就绪'
}

"""
[一个batch

    {一种手语
        三种采集数据
        'acc': [ []...每次采集的数据  ] 
        'gyr': [ []...  ]
        'emg': [ []...  ]
    }....
    
]

一个batch 对应一个 CaptureStore
"""

class CaptureStore:
    def __init__(self):
        # 检查当前是否有上次的缓存 有则录入 无则生成
        try:
            file_ = open('.\\tmp_collection.data', 'r+b')
            tmp_data = pickle.load(file_)
            file_.close()
            # 从tmp文件中展开对象
            self.capture_batch = tmp_data[0]
            # 当前的batch数
            self.capture_data = tmp_data[1]
            # 当前batch采集的所有数据
            self.curr_capture_sign_data = self.capture_data[-1]
            # 当前手语每次采集的数据
        except IOError:
            self.capture_batch = next_batch()
            self.capture_data = []
            self.curr_capture_sign_data = {
                'acc': [],
                'gyr': [],
                'emg': [],
            }
            self.capture_data.append(self.curr_capture_sign_data)

    @property
    def curr_capture_sign_num(self):
        return len(self.capture_data)

    @property
    def curr_capture_times(self):
        return len(self.curr_capture_sign_data['acc'])

    def append_data(self, acc_data, gyr_data, emg_data):
        """
        采集数据的保存是以 [ [] [] [] ... each次数的采集 ] 形式
        每次采集结束后会调用该方法将这次采集数据追加与数据存储对象
        :param acc_data: 本次采集的acc data
        :param gyr_data:
        :param emg_data:
        :return: None
        """
        self.curr_capture_sign_data['acc'].append(acc_data)
        self.curr_capture_sign_data['gyr'].append(gyr_data)
        self.curr_capture_sign_data['emg'].append(emg_data)


    def is_capture_times_satisfy(self):
        return len(self.curr_capture_sign_data['acc']) >= CAPTURE_TIMES

    def is_empty(self):
        return self.curr_capture_times == 0

    def next_sign(self):
        # 当前batch每种手语采集完毕
        while self.curr_capture_sign_num < len(GESTURES_TABLE):
            self.curr_capture_sign_data = {
                'acc': [],
                'gyr': [],
                'emg': [],
            }
            self.capture_data.append(self.curr_capture_sign_data)
            # curr_capture_sign_num 以列表为主 每种手语capture之前都会先push一个dict
            # 因此capture_sign_num 从1开始 与手语标号相同
            # 而手语列表总是从0开始  因此以sign_num访问gesture_table时 要减一
            if GESTURES_SELECTED_LIST[self.curr_capture_sign_num - 1].get() == 1:
                break

        if self.curr_capture_sign_num == len(GESTURES_TABLE):
            self.save_to_file()
            self.capture_batch = next_batch()
            self.capture_data = []
            self.init_curr_capture_sign()

    def init_curr_capture_sign(self):
        while GESTURES_SELECTED_LIST[self.curr_capture_sign_num - 1].get() != 1:
            self.curr_capture_sign_data = {
                'acc': [],
                'gyr': [],
                'emg': [],
            }
            self.capture_data.append(self.curr_capture_sign_data)



    def discard_curr_sign(self):
        self.curr_capture_sign_data = {
            'acc': [],
            'gyr': [],
            'emg': [],
        }
        if len(self.capture_data) >= 1:
            self.capture_data.pop()
            self.capture_data.append(self.curr_capture_sign_data)


    def discard_curr_capture(self):
        if len(self.curr_capture_sign_data['acc']) > 0:
            self.curr_capture_sign_data['acc'].pop()
            self.curr_capture_sign_data['gyr'].pop()
            self.curr_capture_sign_data['emg'].pop()


    def save_to_file(self):
        os.makedirs(DATA_PATH + '\\' + str(self.capture_batch))
        curr_data_path = os.path.join(DATA_PATH, str(self.capture_batch))
        DIR_NAME_LIST = ['Acceleration', 'Emg', 'Gyroscope']
        for i in range(len(TYPE_LIST)):
            dir_name = DIR_NAME_LIST[i]
            data_type = TYPE_LIST[i]
            data_load_path = os.path.join(curr_data_path, dir_name)
            os.makedirs(data_load_path)
            for cap_num in range(len(self.capture_data)):
                file_path = os.path.join(data_load_path, str(cap_num + 1) + '.txt')
                file_ = open(file_path, 'w')
                cap_data = self.capture_data[cap_num][data_type]
                data_lines_str = ''
                for data_block in cap_data:
                    for each_line in data_block:
                        data_lines_str += str(each_line) + '\n'
                file_.write(data_lines_str)
                file_.close()
        try:
            os.remove('.\\tmp_collection.data')
        except WindowsError:
            print('tmp file doesn\'t created no need to delete')


    def save_tmp_obj(self):
        file_ = open('tmp_collection.data', 'w+b')
        store_obj_tmp = (self.capture_batch,
                         self.capture_data)
        pickle.dump(store_obj_tmp, file_)
        file_.close()

    def get_store_status(self):
        res = '当前batch号: %d\n当前手语id: %d %s\n当前采集次数: %d' % \
              (self.capture_batch,
               self.curr_capture_sign_num,
               GESTURES_TABLE[self.curr_capture_sign_num - 1],
               self.curr_capture_times)
        return res

class CaptureControl(object):
    def __init__(self,
                 myo_device,
                 view):
        self.capture_state = STATE_END_OF_CAPTURE
        self._myo_device = myo_device
        self._time = 0
        self._t_s = 0.01
        self.Emg = []
        self.Acc = []
        self.Gyr = []
        self.capture_store = CaptureStore()
        self.curr_capture_tag = self.capture_store.curr_capture_times + 1
        self.view = view
        self.is_cap_discard = False
        self.is_cap_store = False
        self.is_auto_capture = False
        self.is_last_cap = False
        self.update_capture_state_info()

    def start(self):

        while self.capture_state != STATE_STOP_COLLECTION:
            # 一个手势若干遍采集的开始
            wait_time_start = time.time()
            # 等待用户1.5s
            while (time.time() - wait_time_start) < 1.5:
                # 当用户存储,丢弃,暂停采集时退出等待
                if self.capture_state != STATE_STANDBY:
                    break
                if self.is_cap_store or self.is_cap_discard:
                    time.sleep(1)
                    break
            # 如果用户在等待时间什么都没做 帮用户存储
            if not self.is_cap_store and not self.is_cap_discard \
                    and self.capture_state == STATE_STANDBY:
                self.auto_continue_capture()

            if self.capture_state == STATE_START_CAPTURE:
                # 手势每次的采集
                self.capture_state = STATE_CAPTURING
                self.update_capture_state_info()
                self.Acc = []
                self.Gyr = []
                self.Emg = []
                self.is_cap_discard = False
                self.is_cap_store = False

                self._myo_device.vibrate(VibrationType.short)
                time.sleep(0.2)
                start_time = time.clock()
                while self.capture_state == STATE_CAPTURING:
                    current_time = time.clock()
                    # 以固定频率进行采集
                    gap_time = current_time - start_time
                    if gap_time > self._t_s:
                        start_time = time.clock()
                        self.store_data()
        if not self.capture_store.is_empty():
            self.capture_store.save_tmp_obj()

    # 每次采集的长度是通过采集次数限制的
    def cap_data(self):
        next_tag_num = self.capture_store.curr_capture_times
        self.Emg.append(self._myo_device.emg + (next_tag_num,))
        self.Acc.append([it for it in self._myo_device.acceleration])
        self.Gyr.append([it for it in self._myo_device.gyroscope])

    def store_data(self):
        if self.is_capture_data_length_satisfy():
            # 一个手势的160次data capture
            self.capture_state = STATE_STANDBY
            self.update_capture_state_info()
            self._myo_device.vibrate(VibrationType.short)
            if self.capture_store.is_capture_times_satisfy():
                self.is_last_cap = True
                self.capture_store.next_sign()
                self.update_capture_state_info()
        #     进行下一个手语采集
        else:
            self.cap_data()

    def is_capture_data_length_satisfy(self):
        return len(self.Emg) == 180 \
               and len(self.Acc) == 180 \
               and len(self.Emg) == 180

    def start_capture(self):
        if 1 not in [each.get() for each in GESTURES_SELECTED_LIST]:
            return
        else:
            self.capture_store.init_curr_capture_sign()
        self.capture_state = STATE_START_CAPTURE

    def auto_continue_capture(self):
        if not self.is_last_cap:
            self.store_single_capture_data()
        self.is_last_cap = False
        self.is_cap_store = True
        if self.is_auto_capture:
            self.capture_state = STATE_START_CAPTURE
        self.update_capture_state_info()

    def store_single_capture_data(self):
        self.is_cap_store = True
        self.capture_store.append_data(self.Acc, self.Gyr, self.Emg)
        self.update_capture_state_info()
        if self.is_auto_capture:
            self.start_capture()

    def continue_capture(self):
        if not self.is_cap_store and len(self.Acc) == 180:
            self.store_single_capture_data()
        if not self.is_auto_capture:
            self.capture_state = STATE_START_CAPTURE

        self.update_capture_state_info()

    def discard_capture(self):
        if self.capture_state == STATE_STANDBY:
            self.is_cap_discard = True
            if self.is_auto_capture:
                self.start_capture()
            else:
                self.pause_capture()
                self.update_capture_state_info()

    def discard_sign(self):
        self.capture_store.discard_curr_sign()
        self.update_capture_state_info()

    def pause_capture(self):
        self.capture_state = STATE_END_OF_CAPTURE
        self.update_capture_state_info()

    def stop_capture(self):
        self.capture_state = STATE_STOP_COLLECTION
        self.update_capture_state_info()
        self.view.distory_window()
        if self.capture_store.curr_capture_times == 1 \
                and self.capture_store.curr_capture_times == 0:
            num = get_max_batch_num() - 1
            file_ = open('batch_num', 'w')
            file_.write(str(num))
            file_.close()

        # todo debug ,remove
        data = self.capture_store.curr_capture_sign_data
        file_ = open('data_aaa', 'w+b')
        pickle.dump(data, file_)
        file_.close()

    def auto_capture(self):
        self.is_auto_capture = False if self.is_auto_capture else True
        self.capture_state = STATE_END_OF_CAPTURE
        self.update_capture_state_info()

    def update_capture_state_info(self):
        res = self.capture_store.get_store_status()
        res += '\n' + STATE_TABLE[self.capture_state]
        res += '\n自动采集: ' + ('开' if self.is_auto_capture else '关')
        res += '\n'
        if self.is_cap_discard:
            res += '当前采集数据被丢弃'
        if self.is_cap_store:
            res += '当前采集数据被保存'
        self.view.update_info(res)


class ControlPanel:
    def __init__(self, wrap_window):
        self.info_frame = Tkinter.Frame(wrap_window)
        self.info_frame.pack()
        self.info_display = Tkinter.Text(self.info_frame)
        self.info_display.pack()
        self.text = ''
        self.wrap_window = wrap_window

        self.sign_select_frame = Tkinter.Frame(wrap_window)
        self.sign_select_frame.pack(side=TOP)

        self.button_frame = Tkinter.Frame(wrap_window)
        self.button_frame.pack(side=TOP)

        self.init_signs_select_checkbox()


    def set_control(self, capture_control):
        button_start_capture = Tkinter.Button(self.button_frame,
                                              text='开始采集',
                                              command=capture_control.start_capture)
        button_start_capture.pack(side=LEFT)
        button_auto_capture = Tkinter.Button(self.button_frame,
                                             text='自动采集',
                                             command=capture_control.auto_capture)
        button_auto_capture.pack(side=LEFT)
        button_pause_capture = Tkinter.Button(self.button_frame,
                                              text='暂停采集',
                                              command=capture_control.pause_capture)
        button_pause_capture.pack(side=LEFT)
        button_discard_capture = Tkinter.Button(self.button_frame,
                                                text='丢弃此次采集',
                                                command=capture_control.discard_capture)
        button_discard_capture.pack(side=LEFT)
        button_save_capture = Tkinter.Button(self.button_frame,
                                             text='保存此次采集',
                                             command=capture_control.store_single_capture_data)
        button_save_capture.pack(side=LEFT)
        button_save_capture = Tkinter.Button(self.button_frame,
                                             text='删除该手语的采集',
                                             command=capture_control.discard_sign)
        button_save_capture.pack(side=LEFT)

        button = Tkinter.Button(self.button_frame,
                                text="退出",
                                command=capture_control.stop_capture)
        button.pack(side=LEFT)

    def update_info(self, info):
        self.text = info
        self._update_text()

    def _update_text(self):
        self.info_display.delete(1.0, END)
        self.info_display.insert(END, self.text)

    def init_signs_select_checkbox(self):
        for each in range(len(GESTURES_TABLE)):
            var = Tkinter.IntVar()
            GESTURES_SELECTED_LIST[each] = var
            button = Tkinter.Checkbutton(self.sign_select_frame,
                                         text=GESTURES_TABLE[each],
                                         variable=var)
            if each < 7:
                button.grid(row=0, column=each)
            else:
                button.grid(row=1, column=each - 7)

    def distory_window(self):
        self.wrap_window.destroy()

class CaptureThread(threading.Thread):
    def __init__(self, capture_control):
        threading.Thread.__init__(self)
        self.capture_control = capture_control

    def run(self):
        self.capture_control.start()
        return


def get_max_batch_num():
    file = open('batch_num', 'r')
    num = int(file.readline())
    file.close()
    return num

def next_batch():
    file = open('batch_num', 'r')
    num = int(file.readline())
    file.close()
    num += 1
    file = open('batch_num', 'w')
    file.write(str(num) + '\n')
    file.close()
    return num

def main():
    myo.init(os.path.dirname(__file__))
    feed = myo.Feed()
    hub = myo.Hub()
    hub.run(1000, feed)
    myo_device = feed.get_devices()
    print(myo_device)
    time.sleep(1)
    try:
        myo_device = myo_device[0]
        myo_device.set_stream_emg(myo.StreamEmg.enabled)
        wrap_window = Tkinter.Tk()
        wrap_window.title('collect')
        wrap_window.geometry('640x480')
        panel = ControlPanel(wrap_window)
        capture_control = CaptureControl(myo_device, panel)
        panel.set_control(capture_control)

        t = CaptureThread(capture_control)
        t.start()

        wrap_window.mainloop()
    except RuntimeError:
        print('device didn\'t connected')

if __name__ == '__main__':
    main()
