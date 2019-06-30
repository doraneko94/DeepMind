import tkinter as tk

win = tk.Tk()
win.title("闘獣棋")
win.geometry("448x576")

board_img = tk.PhotoImage(file="img/board.png")
playerA_img = [tk.PhotoImage(file="img/mouse.png"), tk.PhotoImage(file="img/cat.png"),
               tk.PhotoImage(file="img/dog.png"), tk.PhotoImage(file="img/wolf.png"),
               tk.PhotoImage(file="img/panther.png"), tk.PhotoImage(file="img/lion.png"),
               tk.PhotoImage(file="img/lion.png"), tk.PhotoImage(file="img/elephant.png")]
playerB_img = [tk.PhotoImage(file="img/mouse2.png"), tk.PhotoImage(file="img/cat2.png"),
               tk.PhotoImage(file="img/dog2.png"), tk.PhotoImage(file="img/wolf2.png"),
               tk.PhotoImage(file="img/panther2.png"), tk.PhotoImage(file="img/lion2.png"),
               tk.PhotoImage(file="img/lion2.png"), tk.PhotoImage(file="img/elephant2.png")]
cv = tk.Canvas(width=448-1, height=576-1)
cv.place(x=0, y=0)
cv.create_image(0, 0, image=board_img, anchor="nw", tag="a")
#cv.create_image(0, 0, image=playerA_img[1], anchor="nw")
cv.delete("b")

import numpy as np
from time import sleep

for i in range(5):
    if np.random.rand() > 0.5:
        cv.create_image(0, 0, image=playerA_img[1], anchor="nw", tag="b")
    else:
        cv.delete("b")
    sleep(5)
    win.update()