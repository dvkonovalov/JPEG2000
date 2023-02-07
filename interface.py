import tkinter as tk
from tkinter import filedialog

global path_image   #путь к файлу обычного изображения


def choice_file_convert():
    global path_image
    path_image = filedialog.askopenfilename()
    print(path_image)


window = tk.Tk()
width = window.winfo_screenwidth()//4*3
height = window.winfo_screenheight()//4*3
window.title("JPEG2000")
window.config(width=width, height=height)
tk.Label(window, text='Конвертировать в JPEG2000', font='Times 24').place(x=width // 3, y=10)
bt = tk.Button(window, text='выбрать файл', font='Times 14', command=choice_file_convert)
bt.place(x=width//20, y=height//8, width=150, height=35)

window.mainloop()