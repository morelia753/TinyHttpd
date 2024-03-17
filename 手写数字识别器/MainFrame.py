import tkinter as tk
import os
from Config import *
from TopBar import TopBar
from CenCanvas import CenCanvas
from StatusBar import StatusBar
from Estimator2 import Estimator2

class MainFrame(tk.Tk):
    def __init__(self):
        super(MainFrame,self).__init__()
        #设置窗口尺寸位置
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        px = int(sw/2-SCREEN_W/2)
        py = int(sh/2-SCREEN_H/2)
        self.geometry(f'{SCREEN_W}x{SCREEN_H}+{px}+{py}')
        # 设置其他属性
        self.resizable(False, False)
        self.title('数字手写板')
        # 初始化其他控件
        self.init_widgets()
        #初始化模型
        from Estimator3 import Estimator3
        self.model = Estimator3()

    #为按钮绑定函数
    def init_widgets(self):
        """初始化窗口控件"""
        callbacks = {
            '清屏': self.clear,
            '保存': self.save,
            '展示': self.show,
            '打开': self.open,
            '训练': self.train,
            '预测': self.predict,
        }
        self.tbar = TopBar(callbacks=callbacks)
        self.tbar.pack(side=tk.TOP, fill=tk.X)
        self.ccav = CenCanvas()
        self.ccav.pack()
        self.sbar = StatusBar()
        self.sbar.pack(side=tk.BOTTOM, fill=tk.X)

    # 下面为按钮绑定函数
    def clear(self):
        print('清屏')
        self.ccav.clear_canvas()

    def save(self):
        print('保存')
        self.ccav.save()

    def show(self):
        print('展开')
        if os.name == 'nt':
            os.startfile(os.path.abspath('./data/train'))
        else:
            subprocess.run(['open', os.path.abspath('./data/train')])

    def open(self):
        print('打开')
        self.ccav.open()

    def train(self):
        print('训练')
        train_acc, test_acc = self.model.train()
        self.sbar.set_status('训练: %.2f 测试: %.2f' % (train_acc, test_acc))

    def predict(self):
        print('推理')
        image = self.ccav.generate_image()
        image.save('data/train/temp.png', 'png')
        label = self.model.predict()
        self.sbar.set_status('预测: %d' % label)