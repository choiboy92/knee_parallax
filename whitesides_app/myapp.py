import tkinter as tk
import os
import random
import numpy as np
from PIL import Image, ImageTk

def drawEpicondylar():
    global item
    item = "epi"
    canvas.bind("<Button-1>", click)
    canvas.bind("<B1-Motion>", drag)
    print("Clicked Draw Epicondylar Axis")
    return

def drawWhitesides():
    print("Clicked Draw Whitesides Axis")
    global item
    item = "white"
    canvas.bind("<Button-1>", click)
    canvas.bind("<B1-Motion>", drag)
    return

def ParallaxProcess():
    print("Clicked ParallaxProcess")
    epicondylar = np.array(canvas.coords("epi")).reshape(2,2)
    epicondylar[:,1] = 400 - epicondylar[:,1]
    print(epicondylar)
    if epicondylar[0,0]>epicondylar[1,0]:
        epicondylar_med = epicondylar[1]
        epicondylar_lat = epicondylar[0]
    else:
        epicondylar_med = epicondylar[0]
        epicondylar_lat = epicondylar[1]
    whitesides = np.array(canvas.coords("white")).reshape(2,2)
    whitesides[:,1] = 400 - whitesides[:,1]
    print(whitesides)
    if whitesides[0,1]>whitesides[1,1]:
        whitesides_ant = whitesides[0]
        whitesides_post = whitesides[1]
    else:
        whitesides_ant = whitesides[1]
        whitesides_post = whitesides[0]
    viewrotations = np.array(name[current_image_num][:-4].split('_'))
    viewrotations = [ int(x) for x in viewrotations ]
    print(viewrotations)
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

def click(e):
    canvas.delete(item)
    if item == "epi":
        c = "green"
    elif item == "white":
        c = "red"
    coords["x"] = e.x
    coords["y"] = e.y

    # create a line on this point and store it in the list
    lines.append(canvas.create_line(coords["x"],coords["y"],coords["x"],coords["y"], fill=c, tag=item))

def drag(e):
    coords["x2"] = e.x
    coords["y2"] = e.y
    # Change the coordinates of the last created line to the new coordinates
    canvas.coords(lines[-1], coords["x"],coords["y"],coords["x2"],coords["y2"])

coords = {"x":0,"y":0,"x2":0,"y2":0}
# keep a reference to all lines by keeping them in a list
lines = []

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
