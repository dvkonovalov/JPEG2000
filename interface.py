import tkinter as tk
from tkinter import filedialog

global path_image   #путь к файлу обычного изображения


def choice_file_convert():
    global path_image
    path_image = filedialog.askopenfilename()
    print(path_image)


window = tk.Tk()
width = 960
height = 640
window.title("JPEG2000")
window.config(width=width, height=height)
tk.Label(window, text='Конвертировать в JPEG2000', font='Times 20').place(x=300, y=10)
bt1 = tk.Button(window, text='выбрать файл', font='Times 14', command=choice_file_convert)
bt1.place(x=15, y=50, width=150, height=35)
stroka_fail = tk.Label(window, text = 'Файл не выбран', font = 'Times 14')
stroka_fail.place(x=200, y = 55)


window.mainloop()