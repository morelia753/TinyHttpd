import tkinter as tk

class StatusBar(tk.Frame):
    def __init__(self):
        super(StatusBar,self).__init__()
        self.config(bg='#d2dae2',height=30)

        self.status = tk.Label(self,text='准备就绪')
        # self.status.config(bg='#d2dae2', height=30)
        self.status.pack(side=tk.LEFT)


    def set_status(self,text):
        self.status['text'] = text



if __name__ == '__main__':
    window = tk.Tk()
    window.geometry('500x575+200+300')
    window.resizable(False,False)

    sbar = StatusBar()
    sbar.pack(side=tk.BOTTOM,fill=tk.X)

    button1 = tk.Button(text="按钮",command=lambda :sbar.set_status('修改文本'))
    button1.pack()

    window.mainloop()
