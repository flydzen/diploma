import tkinter
from pathlib import Path
from cairosvg import svg2png
from utils.svg import SVG

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk


class App:
    blacklist = Path('data/blacklist.txt')
    skip = 1100

    def __init__(self):
        self.fonts = (directory for directory in Path('data/svg').iterdir())
        self.size = len(list(Path('data/svg').iterdir()))
        self.step = 0

        self.font = None

        self.root = tkinter.Tk()

        # создаем рабочую область
        self.frame = tkinter.Frame(self.root)
        self.frame.grid()

        # Добавим метку

        self.label = ttk.Label(self.frame, text='...')
        self.label.grid(column=0, row=0)

        self.btn_accept = ttk.Button(self.frame, text="OK [x]", command=self.accept)
        self.btn_accept.grid(column=0, row=1)

        self.btn_reject = ttk.Button(self.frame, text="Не ОК [z]", command=self.reject)
        self.btn_reject.grid(column=0, row=2)

        self.num = ttk.Label(self.frame, text=f'0/{self.size}')
        self.num.grid(column=0, row=3)

        self.image = Image.open("tmp.png")
        self.photo = ImageTk.PhotoImage(self.image)

        # Добавим изображение
        self.canvas = tkinter.Canvas(self.root, height=600, width=700)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(row=0, column=1)

        self.canvas.bind('x', self.accept)
        self.canvas.bind('z', self.reject)
        self.canvas.focus_set()

        self.accept(None)

        self.root.mainloop()

    def accept(self, event):
        while True:
            font = next(self.fonts)

            self.step += 1
            if self.step < self.skip:
                continue
            self.font = font.name
            self.label.config(text=self.font)
            self.num.config(text=f'{self.step}/{self.size}')

            for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
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

    def reject(self, event):
        with self.blacklist.open('a') as file:
            print(self.font.encode('utf-8'), file=file)
        self.accept(event)


app = App()
