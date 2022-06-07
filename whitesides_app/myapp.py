import tkinter as tk
from tkinter import messagebox, Toplevel
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
            category text,
            view_rotation_flex real,
            view_rotation_varvalg real,
            view_rotation_intext real,
            user_angle real,
            true_proj_angle real,
            user_deviation real,
            epi_deviation real,
            white_deviation real,
            postcond_deviation real,
            whitesides_ant real,
            whitesides_post real,
            epi_med real,
            epi_lat real,
            postcond_med real,
            postcond_lat real
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

def drawPosterior():
    print("Clicked Draw Posterior Condylar")
    global item
    item = "postcond"
    canvas.bind("<Button-1>", click)
    canvas.bind("<B1-Motion>", drag)
    return

# Coordinates of references in drawing plane
whitesides_ant = np.array([-0.4435, 45.8136, -0.0914])/10
whitesides_post = np.array([3.6296, -14.9831, 0.7476])/10
whitesides = whitesides_ant - whitesides_post

epicondylar_med = np.array([44.4236, -0.0000, 9.1508])/10
epicondylar_lat = np.array([-44.4236, 0.0000, -9.1508])/10
epicondylar = epicondylar_med - epicondylar_lat

posteriorcondylar_med = np.array([26.692, 23.254, 6.852])/10
posteriorcondylar_lat = np.array([-28.787, 23.737, 4.3025])/10
posteriorcondylar = posteriorcondylar_med - posteriorcondylar_lat
theta_og = np.abs(np.arccos(np.dot(epicondylar, whitesides)/(np.linalg.norm(epicondylar)*np.linalg.norm(whitesides)))*(180/np.pi))
#print("Original True angle:", theta_og)

def ParallaxProcess():
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

    postcond = np.array(canvas.coords("postcond")).reshape(2,2)
    postcond[:,1] = 400 - postcond[:,1]
    postcond[:,0] = - postcond[:,0]    # Align x-axis with hip coordinate system
    if postcond[0,0]>postcond[1,0]:
        postcond_med = postcond[1]
        postcond_lat = postcond[0]
    else:
        postcond_med = postcond[0]
        postcond_lat = postcond[1]

    epi_vec = epi_lat - epi_med   # So that epicondylar vector aligns with coordinate system
    epi_vec = np.pad(epi_vec, (0,1))
    white_vec = white_ant - white_post
    white_vec = np.pad(white_vec, (0,1))
    theta_user = np.abs(np.arccos(np.dot(epi_vec, white_vec)/(np.linalg.norm(epi_vec)*np.linalg.norm(white_vec)))*(180/np.pi))

    postcond_vec = postcond_lat - postcond_med
    postcond_vec = np.pad(postcond_vec, (0,1))

    viewrotations = np.array(name[current_image_num][:-4].split('_'))
    viewrotations = [ int(x) for x in viewrotations ]
    print("Viewer Rotations:", viewrotations)
    print("User Input angle:",theta_user)



    # Set camera and viewing positions
    camera_d = 300  # in millimetres
    camera = np.array([0,0,camera_d])
    Rview = viewing_plane_transform(viewrotations)
    view_pos = np.matmul(Rview, camera)

    # Project true bone references into viewing plane
    normal = np.zeros(3) - view_pos
    normal_unit = normal/np.linalg.norm(normal)  # unit normal vector

    # Find projected vectors of epicondylar axis and Whitesides onto viewing plane
    proj_epicondylar = np.cross(normal_unit, np.cross(epicondylar, normal_unit))
    proj_whiteside = np.cross(normal_unit, np.cross(whitesides, normal_unit))
    proj_postcond = np.cross(normal_unit, np.cross(posteriorcondylar, normal_unit))
    # Make projection flat in viewing plane
    proj_epicondylar[2] = 0
    proj_whiteside[2] = 0
    proj_postcond[2] = 0

    # Calculate angle between projected vectors
    theta_proj = np.abs(np.arccos(np.dot(proj_epicondylar, proj_whiteside)/(np.linalg.norm(proj_epicondylar)*np.linalg.norm(proj_whiteside)))*(180/np.pi))
    print("True Projected angle:", theta_proj)
    print("User deviation:", np.abs(theta_proj - theta_user))

    theta_epicondylar = np.arccos(np.dot(proj_epicondylar, epi_vec)/(np.linalg.norm(proj_epicondylar)*np.linalg.norm(epi_vec)))*(180/np.pi)
    theta_whitesides = np.arccos(np.dot(proj_whiteside, white_vec)/(np.linalg.norm(proj_whiteside)*np.linalg.norm(white_vec)))*(180/np.pi)
    theta_postcond = np.arccos(np.dot(proj_postcond, postcond_vec)/(np.linalg.norm(proj_postcond)*np.linalg.norm(postcond_vec)))*(180/np.pi)
    #print(proj_postcond)
    #print(postcond_vec)
    #print(np.arctan(proj_postcond[1]/proj_postcond[0])*(180/np.pi))
    #print(np.arctan(postcond_vec[1]/postcond_vec[0])*(180/np.pi))
    print("Epicondylar deviation:", theta_epicondylar)
    print("Whitesides deviation:", theta_whitesides)
    print("Posterior Condylar deviation:", theta_postcond)

    # Create a database or connect to bone
    conn = sqlite3.connect('parallax_data.db')
    # Create cursor (way to make changes to database)
    cursor = conn.cursor()
    # send data to db
    cursor.execute("""INSERT INTO parallax_data VALUES (
                        :name,
                        :category,
                        :view_rotation_flex,
                        :view_rotation_varvalg,
                        :view_rotation_intext,
                        :user_angle,
                        :true_proj_angle,
                        :user_deviation,
                        :epi_deviation,
                        :white_deviation,
                        :postcond_deviation,
                        :whitesides_ant,
                        :whitesides_post,
                        :epi_med,
                        :epi_lat,
                        :postcond_med,
                        :postcond_lat
                        )""",
                    {
                        'name': name_str,
                        'category': category[current_image_num],
                        'view_rotation_flex': viewrotations[0],
                        'view_rotation_varvalg': viewrotations[1],
                        'view_rotation_intext': viewrotations[2],
                        'user_angle': theta_user,                   # user input of angle between whitesides & epicondylar
                        'true_proj_angle': theta_proj,              # real angle between true whitesides & epicondylar projection
                        'user_deviation': np.abs(theta_proj - theta_user),  # error between user input & real whitesides-epicondylar angle
                        'epi_deviation': theta_epicondylar,     # user drawn epicondylar angle error
                        'white_deviation': theta_whitesides,    # user drawn whitesides angle error
                        'postcond_deviation': theta_postcond,    # user drawn posteriorcondylar angle error

                        # Store coordinates of user input positions
                        'whitesides_ant': white_ant,
                        'whitesides_post': white_post,
                        'epi_med': epi_med,
                        'epi_lat': epi_lat,
                        'postcond_med': postcond_med,
                        'postcond_lat': postcond_lat
                    }
    )
    conn.commit()
    conn.close()

    # Update image
    canvas.itemconfig(image_cont, image=chooseImage())
    return

window = tk.Tk()
label = tk.Label(window, text = 'Do not resize window')
label.pack()

# add name entry popup
name_popup = Toplevel(window)
name_popup.attributes('-topmost', 'true')
x = window.winfo_x()
y = window.winfo_y()
name_popup.geometry("+%d+%d" % (x + 400, y + 200))
name_popup.wm_title("Before you start")
popup_label = tk.Label(name_popup, text = 'Enter your name:')
popup_label.pack()
name_popup.geometry("250x150")
name_entry = tk.Entry(name_popup, width= 25)
name_entry.pack()
popup_button= tk.Button(name_popup, text="Ok", command=lambda:popup_button_click(name_popup, name_entry))
popup_button.pack(pady=5)

# function to carry out after name has been entered
def popup_button_click(name_popup, name_entry):
    global name_str
    window.attributes('-topmost', 'true')
    name_str = name_entry.get();
    name_popup.destroy()


def chooseImage():
    global current_image_num
    global current_iter_num
    global iter_num
    global name
    global category
    global index
    # next image
    current_image_num += 1

    # return to first image
    if current_image_num == test_num*2:
        current_iter_num = current_iter_num + 1
        if current_iter_num == iter_num:
            FinishSequence()
        else:
            print(name, category,index)
            temp = list(zip(name, category, index))
            namenew, categorynew, indexnew  = zip(*temp)
            # reassign shuffled imageset
            name = np.array(namenew)
            category = np.array(categorynew)
            index = np.array(indexnew)
            current_image_num = 0
    image_cont = canvas.create_image(0, 0, anchor=tk.NW, image=images[index[current_image_num]])

    return

def collectImages(file, folder):
    img = Image.open(folder+"/"+file)
    img = ImageTk.PhotoImage(img.resize((1000, 550)))
    return img

def click(e):
    canvas.delete(item)
    if item == "epi":
        c = "green"
    elif item == "white":
        c = "red"
    elif item == "postcond":
        c = "blue"
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

# end program after all iterations have been completed
def FinishSequence():
    tk.messagebox.showinfo("You're done",  "Experiment complete! Click to end!")
    #window.quit()
    window.destroy()


coords = {"x":0,"y":0,"x2":0,"y2":0}
# keep a reference to all lines by keeping them in a list
lines = []

canvas = tk.Canvas(window)
canvas.pack(expand = 1, fill=tk.BOTH) # Stretch canvas to root window size.


#render_folder = ["./renders_TKA", "./renders_PKA"]
render_folder = ["./subject_subset_renders/renders_TKA", "./subject_subset_renders/renders_PKA"]

# Pre-load images to use
iter_num = 3
current_iter_num = 0
test_num = 5
images = []
name = []
category = []
index = np.arange(0,test_num*2,1)
for folder in render_folder:
    for i in range(0, test_num):
        chosen = os.listdir(folder)[i]
        images += [collectImages(chosen, folder)]
        name += [chosen]
        category += [folder[-3:]]
# Set first initial image
current_image_num = 0
image_cont = canvas.create_image(0, 0, anchor=tk.NW, image=images[index[current_image_num]])


B1 = tk.Button(window, text ="Draw Epicondylar axis", command=drawEpicondylar)
B1.place(x=50,y=650)
B2 = tk.Button(window, text ="Draw Whitesides", command=drawWhitesides)
B2.place(x=300,y=650)
B3 = tk.Button(window, text ="Draw Posterior Condylar", command=drawPosterior)
B3.place(x=500,y=650)
B4 = tk.Button(window, text ="Go!", fg="green", command = ParallaxProcess)
B4.place(x=900,y=650)

window.title('My App')
window.geometry('1000x700')
window.mainloop()
