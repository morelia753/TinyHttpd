import os.path
import tkinter as tk
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from PIL import Image, ImageTk, ImageDraw

class CenCanvas(tk.Canvas):

    def __init__(self):
        super(CenCanvas, self).__init__()
        self.config(highlightthickness=0, bg='white', width=500, height=500)
        # 显示绘图区域
        self.draw_area()
        #绑定事件
        self.bind_events()
        #存储绘制位置
        self.px = None
        self.py = None
        self.image_hold = None
        self.image_item = None
        #存储数字编号
        self.items = []
        # 清空数字
        self.clear_canvas()
        #打开文件夹
        # self.open()

    def bind_events(self):
        self.bind('<B1-Motion>',self.mouse_motion)
        self.bind('<ButtonRelease-1>',self.mouse_release)

    def mouse_release(self,event):
        self.px = None
        self.px = None

    def mouse_motion(self,event):
        cx = event.x
        cy = event.y
        radius = 4
        draw_points = [(cx, cy)]

        #计算插值
        if self.px and self.py:
            num_steps = int(max(abs(self.px - cx), abs(self.py-cy)) / 2)
            if num_steps > 1:
                points = self.insert_point((self.px, self.py), (cx, cy), num_steps)
                draw_points.extend(points)
        #绘制点
        for point in draw_points:
              tx, ty = point
              item = self.create_oval(tx-radius,ty-radius,tx+radius,ty+radius,fill='red',outline='red')
              self.items.append(item)
        #把当前坐标作为上一次坐标点
        self.px, self.py = cx, cy

    def insert_point(self, point1, point2, num_steps):
        x1, y1 = point1
        x2, y2 = point2
        step_x = (x2 - x1) / num_steps
        step_y = (y2 - y1) / num_steps
        points = []
        for index in range(num_steps):
            ix = x1 + index * step_x
            iy = y1 + index * step_y
            points.append((ix, iy))
        return points

    #绘画
    def draw_area(self):
        width = 300
        height = 300
        x1, y1 = int(500 / 2 - width / 2), int(500 / 2 - height / 2)
        x2, y2 = x1 + width, y1 + height
        self.area = self.create_rectangle(x1, y1, x2, y2, outline="gray", width=2, dash=(15, 15))
    #清空
    def clear_canvas(self):
        for item in self.items:
            self.delete(item)
        self.items.clear()
    #打开图像
    def open(self):
        # 1. 显示选择图像的对话框（图片路径）
        default_path = os.path.abspath('data/train')
        filename = askopenfilename(initialdir=default_path)
        if not filename:
            return
        # 2. 读取图像数据，并进行转换
        image = Image.open(filename)
        photo = ImageTk.PhotoImage(image)
        # 持有打开的图像，避免销毁
        self.image_hold = photo
        self.image_data = image
        # 3. 将转换后的图像数据绘制到画布上
        if hasattr(self, 'image_item') and self.image_item is not None:
            self.delete(self.image_item)
        self.image_item = self.create_image(0,0,image=photo,anchor=tk.NW)
        print(self.image_item)
        self.items.append(self.image_item)

    def generate_image(self):
        #创建空的和画布同等大小的图片
        image = Image.new('RGB',(500,500),self['bg'])
        draw = ImageDraw.Draw(image)
        for item in self.items:
            if self.type(item) == 'image':
                image.paste(self.image_data,box=self.bbox(item))
            else:
                point = self.coords(item)
                color = self.itemcget(item,'fill')
                draw.ellipse(point, fill=color)
        #返回新图像
        return image

    def save(self):
        save_path = os.path.abspath('data/train')
        save_path = asksaveasfilename(initialdir=save_path, defaultextension=".png")
        if not save_path:
            return
        #重绘一副新的图像
        image = self.generate_image()
        #直接保存
        image.save(save_path,"png")


if __name__ == '__main__':
    window = tk.Tk()
    window.geometry('500x575+200+200')
    window.title("手写数字识别器")
    window.resizable('False','False')

    canvas = CenCanvas()
    canvas.pack()

    frame = tk.Frame()
    clear_button = tk.Button(frame,text='清空',command=lambda : canvas.clear_canvas())
    clear_button.pack(side=tk.LEFT)
    open_button = tk.Button(frame, text='打开', command=lambda: canvas.open())
    open_button.pack(side=tk.LEFT)
    save_button = tk.Button(frame, text='保存', command=lambda: canvas.save())
    save_button.pack(side=tk.LEFT)
    frame.pack(side=tk.BOTTOM)
    window.mainloop()
