import tkinter as tk


class TopBar(tk.Frame):

    def __init__(self, callbacks={}):
        super(TopBar, self).__init__()
        self.callbacks = callbacks
        self.init_widgets()

    def init_widgets(self):
        # 加载图标素材
        self.load_images(scale=12)
        # 创建按钮对象
        self.buttons = {}
        for name, image in self.images.items():
            self.buttons[name] = tk.Button(self, text=name,command=self.callbacks.get(name, None))
            self.buttons[name].pack(side=tk.LEFT)

    def load_images(self, scale):
        fnames = (('清屏',     'source/clear.png'),
                  ('保存',      'source/save.png'),
                  ('展示',      'source/show.png'),
                  ('打开',      'source/open.png'),
                  ('训练',     'source/train.png'),
                  ('预测',   'source/predict.png'))
        self.images = {}
        for name, fname in fnames:
            self.images[name] = tk.PhotoImage(file=fname).subsample(scale, scale)


if __name__ == '__main__':

    window = tk.Tk()
    window.geometry('500x575+200+200')
    window.resizable(False, False)

    callbacks = {
        '清屏': lambda : print('清屏'),
        '保存': lambda : print('保存'),
        '展示': lambda : print('展示'),
        '打开': lambda : print('打开'),
        '训练': lambda : print('训练'),
        '预测': lambda : print('预测'),
    }

    topbar = TopBar(callbacks=callbacks)
    topbar.pack(side=tk.TOP, fill=tk.X)


    window.mainloop()