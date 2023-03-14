import tkinter
import tkinter as tk
from tkinter import filedialog
import main

global path_image, path_show, path_convert   #путь к файлу jpg изображения

def choice_image_show():
    global path_show
    result_string.config(text = 'Процесс не запущен', foreground = 'red')
    path_show = filedialog.askopenfilename()
    stroka_file1.config(text='Выбран файл - ' + path_show[path_show.rfind('/') + 1:])


def choice_file_convert():
    global path_image
    result_string.config(text = 'Процесс не запущен', foreground = 'red')
    path_image = filedialog.askopenfilename()
    stroka_file.config(text = 'Выбран файл - ' + path_image[path_image.rfind('/')+1:])


def start_convert_to_jpeg2000():
    try:
        koef = float(quantize_koef.get())
    except:
        koef = 0.1
    if koef>1:
        koef = 1
    if koef<=0:
        koef = 0.1
    main.convert_to_JPEG(path_image, path_image[:path_image.rfind('/')] + '/file.jpeg2000', koef, transform_wavelet)
    stroka_file.config(text = 'Изображение успешно сохранен в ' + path_image[:path_image.rfind('/')] + '/file.jpeg2000')
    result_string.config(text = 'Конвертация в JPEG2000 завершена', foreground = '#00A600')

def show_image():
    main.show_image(path_show)
    result_string.config(text = 'Изображение открыто', foreground = '#00A600')


def choice_convert_image():
    global path_convert
    result_string.config(text = 'Процесс не запущен', foreground = 'red')
    path_convert = filedialog.askopenfilename()
    stroka_file2.config(text = 'Выбран файл - ' + path_convert[path_convert.rfind('/')+1:])

def convert_image():
    main.convert_image(path_convert, path_convert[:path_convert.rfind('/')] + '/image.jpg')
    result_string.config(text = 'Конвертация в JPG завершена', foreground = '#00A600')


window = tk.Tk()
width = 960
height = 640
window.title("JPEG2000")
window.config(width=width, height=height)
tk.Label(window, text='Конвертировать в JPEG2000', font='Times 20').place(x=300, y=10)
bt1 = tk.Button(window, text='выбрать файл', font='Times 14', command=choice_file_convert)
bt1.place(x=10, y=50, width=150, height=35)
stroka_file = tk.Label(window, text = 'Файл не выбран', font = 'Times 14')
stroka_file.place(x=200, y = 55)
stroka_koef = tk.Label(window, text = 'Введите коэффициент сжатия от 0 до 1:', font = 'Times 14')
stroka_koef.place(x = 10, y = 95)
quantize_koef = tk.StringVar()
pole_koef = tk.Entry(font = 'Times 14', textvariable = quantize_koef)
pole_koef.place(x = 350, y = 98)
transform_wavelet = tk.BooleanVar
checkbox_transform = tk.Checkbutton(window, text = 'Максимальное сжатие при данном коэффициенте', font = 'Times 14',
                                    variable = transform_wavelet, onvalue = True, offvalue = False)
checkbox_transform.place(x = 10, y = 125)
start_conv_to_jpeg2000 = tk.Button(window, text = 'Конвертировать в Jpeg2000',activebackground='red', font = 'Times 16', background = '#00A600', command = start_convert_to_jpeg2000)
start_conv_to_jpeg2000.place(x=650, y = 80)



tk.Label(window, text='Посмотреть изображение JPEG2000', font='Times 20').place(x=260, y=180)
stroka_file1 = tk.Label(window, text = 'Файл не выбран', font = 'Times 14')
stroka_file1.place(x=200, y = 235)
bt2 = tk.Button(window, text='выбрать файл', font='Times 14', command=choice_image_show)
bt2.place(x=10, y=230, width=150, height=35)
show_image_jpeg2000 = tk.Button(window, text = 'Посмотреть изображение',activebackground='red', font = 'Times 16', background = '#00A600', command = show_image)
show_image_jpeg2000.place(x=650, y = 230)

tk.Label(window, text='Конвертировать из JPEG2000 в JPG', font='Times 20').place(x=260, y=330)
stroka_file2 = tk.Label(window, text = 'Файл не выбран', font = 'Times 14')
stroka_file2.place(x=200, y = 390)
bt3 = tk.Button(window, text='выбрать файл', font='Times 14', command=choice_convert_image)
bt3.place(x=10, y=385, width=150, height=35)
show_image_jpeg2000 = tk.Button(window, text = 'Конвертировать в JPG',activebackground='red', font = 'Times 16', background = '#00A600', command = convert_image)
show_image_jpeg2000.place(x=650, y = 380)

result_string = tk.Label(window, text = 'Процесс не запущен', font = 'Times 18', foreground = 'red')
result_string.place(x = width//3+35, y = 480)

canvas1 = tkinter.Canvas(bg = 'white', width=width, height=10)
canvas1.place(x = 0, y = 160)
canvas1.create_line(0, 5, width, 5, activefill="red", fill="black", width = 10)
canvas2 = tkinter.Canvas(bg = 'white', width=width, height=10)
canvas2.place(x = 0, y = 310)
canvas2.create_line(0, 5, width, 5, activefill="red", fill="black", width = 10)


window.mainloop()