import tkinter as tk
import os
import random
from PIL import Image, ImageTk

def drawEpicondylar():
    print("Clicked Draw Epicondylar Axis")
    return

def drawWhitesides():
    print("Clicked Draw Whitesides Axis")
    return

def ParallaxProcess():
    print("Clicked ParallaxProcess")
    canvas.itemconfig(image_cont, image=chooseImage())
    return

window = tk.Tk()
label = tk.Label(window, text = 'Hello world')
label.pack()


render_folder=os.listdir("../renders")

def chooseImage():
    global current_image_num

    # next image
    current_image_num += 1

    # return to first image
    if current_image_num == len(images):
        current_image_num = 0
    print(name[current_image_num])
    image_cont = canvas.create_image(0, 0, anchor=tk.NW, image=images[current_image_num])
    return

def collectImages(file):
    img = Image.open("../renders/"+file)
    img = ImageTk.PhotoImage(img.resize((800, 400)))
    return img

canvas = tk.Canvas(window)
canvas.pack(expand = 1, fill=tk.BOTH) # Stretch canvas to root window size.

# Pre-load images to use
test_num = 5
images = []
name = []
for i in range(0, test_num):
    chosen = random.choice(render_folder)
    images += [collectImages(chosen)]
    name += [chosen]

# Set first initial image
current_image_num = 0
print(name[current_image_num])
image_cont = canvas.create_image(0, 0, anchor=tk.NW, image=images[current_image_num])


B1 = tk.Button(window, text ="Draw Epicondylar axis", command=drawEpicondylar)
B1.place(x=50,y=450)
B2 = tk.Button(window, text ="Draw Whitesides", command=drawWhitesides)
B2.place(x=300,y=450)
B3 = tk.Button(window, text ="Go!", fg="green", command = ParallaxProcess)
B3.place(x=600,y=450)

window.title('My App')
window.geometry('800x500')
window.mainloop()
