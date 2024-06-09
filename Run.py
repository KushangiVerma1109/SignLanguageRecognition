import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import PIL
from PIL import ImageTk
import PIL.Image
import pandas as pd
from tkinter import *
import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
#from tkinter import filedialog
import tkinter.messagebox as tm
import main as signtotext
#import videomaker as texttosign
import generate_data as gd
import cnn as train
import os
from itertools import count
import string
from tkinter import *
import time
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from itertools import count, cycle


bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"
class ImageLabel(tk.Label):
    """
    A Label that displays images, and plays them if they are gifs
    :im: A PIL Image instance or a string filename
    """
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        frames = []
 
        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
            
        except EOFError:
            pass
        self.frames = cycle(frames)
 
        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100
 
        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()
 
    def unload(self):
        self.config(image=None)
        self.frames = None
 
    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)
gif_frames=[]
def give_char():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('tmp1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       print(result)
       chars="ABCDEFGHIJKMNOPQRSTUVWXYZ"
       indx=  np.argmax(result[0])
       print(indx)
       return(chars[indx])

def check_sim(i,file_map):
       for item in file_map:
              for word in file_map[item]:
                     if(i==word):
                            return 1,item
       return -1,""

op_dest="C:/Users/raksh/Downloads/Projectcode/Signlanguage/filtered_data/"
alpha_dest="C:/Users/raksh/Downloads/Projectcode/Signlanguage/alphabet/"
dirListing = os.listdir(op_dest)
editFiles = []
for item in dirListing:
       if ".webp" in item:
              editFiles.append(item)

file_map={}
for i in editFiles:
       tmp=i.replace(".webp","")
       #print(tmp)
       tmp=tmp.split()
       file_map[i]=tmp
def func(a):
       all_frames=[]
       final= PIL.Image.new('RGB', (380, 260))
       words=a.split()
       for i in words:
              flag,sim=check_sim(i,file_map)
              if(flag==-1):
                     for j in i:
                            print(j)
                            im = PIL.Image.open(alpha_dest+str(j).lower()+"_small.gif")
                            frameCnt = im.n_frames
                            for frame_cnt in range(frameCnt):
                                   im.seek(frame_cnt)
                                   im.save("tmp.png")
                                   img = cv2.imread("tmp.png")
                                   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                   img = cv2.resize(img, (380,260))
                                   im_arr = PIL.Image.fromarray(img)
                                   for itr in range(15):
                                          all_frames.append(im_arr)
              else:
                     print(sim)
                     im = PIL.Image.open(op_dest+sim)
                     im.info.pop('background', None)
                     im.save('tmp.gif', 'gif', save_all=True)
                     im = PIL.Image.open("tmp.gif")
                     frameCnt = im.n_frames
                     for frame_cnt in range(frameCnt):
                            im.seek(frame_cnt)
                            im.save("tmp.png")
                            img = cv2.imread("tmp.png")
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (380,260))
                            im_arr = PIL.Image.fromarray(img)
                            all_frames.append(im_arr)
       final.save("out.gif", save_all=True, append_images=all_frames, duration=100, loop=0)
       return all_frames      

img_counter = 0
img_text=''

def Home():
        global window
        
        
        def clear():
                
            print("Clear1")
            txt.delete(0, 'end')
            txt1.delete(0, 'end') 
            
        def sel():
            selection = str(var.get())
            label.config(text = selection)
            

        window = tk.Tk()
        var = IntVar()
        window.title("Sign Language Translation")
##        bg = PhotoImage(file = "1.png")
##        label1 = Label( window, image = bg)
##        label1.place(x = 0, y = 0)
        

 
        window.geometry('1280x720')
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="Sign Language Translation" ,bg=bgcolor  ,fg=fgcolor  ,width=70  ,height=2,font=('times', 25, 'italic bold underline')) 
        message1.place(x=50, y=10)

        lbl = tk.Label(window, text="Enter Your Text",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl.place(x=100, y=100)
        
        txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt.place(x=400, y=115)
        lbl1 = tk.Label(window, text="Enter Your Caption",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1.place(x=100, y=250)
        
        txt1 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1.place(x=400, y=265)

        
        R1 = Radiobutton(window, text="Sign2Text", variable=var, value=1,command=sel)
        R1.place(x=350, y=160)
        #R1.pack( anchor = W )
        R2 = Radiobutton(window, text="Text2Sign", variable=var, value=2,command=sel)
        R2.place(x=450, y=160)
       
        label = tk.Label(window, text="",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        label.place(x=350, y=210)
        lbl = ImageLabel(window)
        lbl.place(x=350, y=350)
        


       

        


        def sign2text():
                lang=label.cget("text")
                              
                
                if lang=="1":
                        signtotext.process()
                        
                else:
                        text=txt.get()
                        if text!="":
                                
                                
                                
                                gif_frames=func(text)
                                print("Len of gif frames==",len(gif_frames))
                                lbl.load('out.gif')
                                #time.sleep(15)
                                #lbl.unload()
                                
                                
                                
                        else:
                                tm.showinfo("Input","Please enter text to Get gif")
        def datacreation():
                text=txt1.get()
                if text!="":
                        gd.process(text)
                else:
                         tm.showinfo("Error","Enter the Caption Letter")
                        
        def trainprocess():
                train.process()
                
                
                        
                


        

        browse = tk.Button(window, text="Start", command=sign2text  ,fg=fgcolor  ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse.place(x=650, y=110)


        clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=15  ,height=1 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton.place(x=320, y=350)

        gen = tk.Button(window, text="Data Creation", command=datacreation  ,fg=fgcolor  ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        gen.place(x=650, y=265)
        trainb = tk.Button(window, text="train", command=trainprocess  ,fg=fgcolor  ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        trainb.place(x=650, y=350)
        

        quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=990, y=350)

        window.mainloop()
Home()

