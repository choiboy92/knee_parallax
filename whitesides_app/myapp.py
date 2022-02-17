import tkinter as tk
import os
import random
import numpy as np
from PIL import Image, ImageTk
import sqlite3

# Create a database or connect to bone
conn = sqlite3.connect('parallax_data.db')

# Create cursor (way to make changes to database)
cursor = conn.cursor()

# create table if not already exists
cursor.execute("""CREATE TABLE IF NOT EXISTS parallax_data (
            name text,
            view_rotation_flex real,
            view_rotation_varvalg real,
            view_rotation_intext real,
            user_angle real,
            true_proj_angle real,
            user_deviation real,
            epi_deviation real,
            white_deviation real
            )""")

conn.commit()   # commit changes
conn.close()    # close connection

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
    epi = np.array(canvas.coords("epi")).reshape(2,2)
    epi[:,1] = 400 - epi[:,1]
    epi[:,0] = - epi[:,0]    # Align x-axis with hip coordinate system

    if epi[0,0]>epi[1,0]:
        epi_med = epi[1]
        epi_lat = epi[0]
    else:
        epi_med = epi[0]
        epi_lat = epi[1]
    white = np.array(canvas.coords("white")).reshape(2,2)
    white[:,1] = 400 - white[:,1]
    white[:,0] = - white[:,0]    # Align x-axis with hip coordinate system
    if white[0,1]>white[1,1]:
        white_ant = white[0]
        white_post = white[1]
    else:
        white_ant = white[1]
        white_post = white[0]

    epi_vec = epi_lat - epi_med   # So that epicondylar vector aligns with coordinate system
    epi_vec = np.pad(epi_vec, (0,1))
    white_vec = white_ant - white_post
    white_vec = np.pad(white_vec, (0,1))
    theta_user = np.abs(np.arccos(np.dot(epi_vec, white_vec)/(np.linalg.norm(epi_vec)*np.linalg.norm(white_vec)))*(180/np.pi))


    viewrotations = np.array(name[current_image_num][:-4].split('_'))
    viewrotations = [ int(x) for x in viewrotations ]
    print("Viewer Rotations:", viewrotations)
    print("User Input angle:",theta_user)

    whitesides_ant = np.array([-0.4435, 45.8136, -0.0914])/10
    whitesides_post = np.array([3.6296, -14.9831, 0.7476])/10
    whitesides = whitesides_ant - whitesides_post

    epicondylar_med = np.array([44.4236, -0.0000, 9.1508])/10
    epicondylar_lat = np.array([-44.4236, 0.0000, -9.1508])/10
    epicondylar = epicondylar_med - epicondylar_lat

    theta_og = np.abs(np.arccos(np.dot(epicondylar, whitesides)/(np.linalg.norm(epicondylar)*np.linalg.norm(whitesides)))*(180/np.pi))
    #print("Original True angle:", theta_og)

    # Set camera and viewing positions
    camera_d = 200/10
    camera = np.array([0,0,camera_d])
    Rview = viewing_plane_transform(viewrotations)
    view_pos = np.matmul(Rview, camera)

    # Project true bone references into viewing plane
    normal = np.zeros(3) - view_pos
    normal_unit = normal/np.linalg.norm(normal)  # unit normal vector

    # Find projected vectors of epicondylar axis and Whitesides onto viewing plane
    proj_epicondylar = np.cross(normal_unit, np.cross(epicondylar, normal_unit))
    proj_whiteside = np.cross(normal_unit, np.cross(whitesides, normal_unit))
    # Make projection flat in viewing plane
    #proj_epicondylar[2] = 0
    #proj_whiteside[2] = 0

    # Calculate angle between projected vectors
    theta_proj = np.abs(np.arccos(np.dot(proj_epicondylar, proj_whiteside)/(np.linalg.norm(proj_epicondylar)*np.linalg.norm(proj_whiteside)))*(180/np.pi))
    print("True Projected angle:", theta_proj)
    print("User deviation:", np.abs(theta_proj - theta_user))

    theta_epicondylar = np.abs(np.arccos(np.dot(proj_epicondylar, epi_vec)/(np.linalg.norm(proj_epicondylar)*np.linalg.norm(epi_vec)))*(180/np.pi))
    theta_whitesides = np.abs(np.arccos(np.dot(proj_whiteside, white_vec)/(np.linalg.norm(proj_whiteside)*np.linalg.norm(white_vec)))*(180/np.pi))

    print("Epicondylar deviation:", theta_epicondylar)
    print("Whitesides deviation:", theta_epicondylar)
    canvas.itemconfig(image_cont, image=chooseImage())

    # Create a database or connect to bone
    conn = sqlite3.connect('parallax_data.db')
    # Create cursor (way to make changes to database)
    cursor = conn.cursor()
    # send data to db
    cursor.execute("""INSERT INTO parallax_data VALUES (
                        :name,
                        :view_rotation_flex,
                        :view_rotation_varvalg,
                        :view_rotation_intext,
                        :user_angle,
                        :true_proj_angle,
                        :user_deviation,
                        :epi_deviation,
                        :white_deviation
                        )""",
                    {
                        'name': name_entry.get(),
                        'view_rotation_flex': viewrotations[0],
                        'view_rotation_varvalg': viewrotations[1],
                        'view_rotation_intext': viewrotations[2],
                        'user_angle': theta_user,
                        'true_proj_angle': theta_proj,
                        'user_deviation': np.abs(theta_proj - theta_user),
                        'epi_deviation': theta_epicondylar,
                        'white_deviation': theta_epicondylar
                    }
    )
    conn.commit()
    conn.close()
    return

window = tk.Tk()
label = tk.Label(window, text = 'Do not resize window')
label.pack()

# add name entry
name_label = tk.Label(window, text="Name:")
name_entry = tk.Entry(window)
name_label.place(x=550,y=0)
name_entry.place(x=600,y=0)


render_folder=os.listdir("../renders")

def chooseImage():
    global current_image_num

    # next image
    current_image_num += 1

    # return to first image
    if current_image_num == len(images):
        current_image_num = 0
    #print(name[current_image_num])
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

def viewing_plane_transform(angle_val):
  flexion = angle_val[0]*np.pi/180
  var_val = (angle_val[1]*np.pi/180) + np.pi
  int_ext = (angle_val[2]*np.pi/180) + np.pi/2

  Rx = np.array([[1, 0, 0],
                [0, np.cos(flexion), -np.sin(flexion)],
                [0, np.sin(flexion), np.cos(flexion)]])
  Ry = np.array([[np.cos(var_val), 0, np.sin(var_val)],
                [0, 1, 0],
                [-np.sin(var_val), 0, np.cos(var_val)]])
  Rz = np.array([[np.cos(int_ext), -np.sin(int_ext), 0],
                [np.sin(int_ext), np.cos(int_ext), 0],
                [0, 0, 1]])
  R = Rx.dot(Ry).dot(Rz)
  return R



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
