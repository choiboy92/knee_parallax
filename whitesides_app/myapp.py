import tkinter as tk

window = tk.Tk()
label = tk.Label(window, text = 'Hello world')
label.pack()

def helloCallBack():
   tkMessageBox.showinfo( "Hello Python", "Hello World")

B = tk.Button(window, text ="Hello", command = helloCallBack)
B.pack()
window.title('My App')
window.geometry('800x500')
window.mainloop()
