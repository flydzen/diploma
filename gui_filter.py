import tkinter
from pathlib import Path
from cairosvg import svg2png
from utils.svg import SVG

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk


class App:
    blacklist = Path('data/blacklist2.txt')
    skip = 0

    def __init__(self):
        self.fonts = [directory for directory in Path('data/svg').iterdir()
                      if Path(f'data/dafonts-free-v1/fonts/{directory.name}.otf').exists()]
        self.size = len(self.fonts)
        self.step = 0

        self.font = None

        self.root = tkinter.Tk()

        # создаем рабочую область
        self.frame = tkinter.Frame(self.root)
        self.frame.grid()

        # Добавим метку

        self.label = ttk.Label(self.frame, text='...')
        self.label.grid(column=0, row=0)

        self.btn_back = ttk.Button(self.frame, text="BACK [c]", command=self.accept)
        self.btn_back.grid(column=0, row=1)

        self.btn_accept = ttk.Button(self.frame, text="OK [x]", command=self.accept)
        self.btn_accept.grid(column=0, row=2)

        self.btn_reject = ttk.Button(self.frame, text="Не ОК [z]", command=self.reject)
        self.btn_reject.grid(column=0, row=3)

        self.num = ttk.Label(self.frame, text=f'0/{self.size}')
        self.num.grid(column=0, row=4)

        self.image = Image.open("tmp.png")
        self.photo = ImageTk.PhotoImage(self.image)

        # Добавим изображение
        self.canvas = tkinter.Canvas(self.root, height=256, width=256)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(row=0, column=1)

        self.canvas.bind('x', self.accept)
        self.canvas.bind('z', self.reject)
        self.canvas.focus_set()

        self.accept(None)

        self.root.mainloop()

    def accept(self, event):
        while self.step < len(self.fonts):
            font = self.fonts[self.step]

            if self.step < self.skip:
                continue
            self.font = font.name
            self.label.config(text=self.font)

            self.num.config(text=f'{self.step}/{self.size}')

            self.step += 1

            for letter in ['A', 'one', '_moreSVGs_', 'B', 'two', 'C', 'three', 'g']:
                file = font / f'{letter}.svg'
                if file.exists():
                    break
            else:
                continue
            svg = SVG.load(file)
            svg.mulsize(255)
            svg2png(bytestring=svg.dump(), write_to="tmp.png")
            self.image = Image.open("tmp.png")
            self.photo = ImageTk.PhotoImage(self.image)
            self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
            break

    def back(self, event):
        self.step -= 2
        self.accept(event)

    def reject(self, event):
        with self.blacklist.open('a', encoding='utf-8') as file:
            file.write(self.font)
            file.write('\n')
        self.accept(event)


app = App()
