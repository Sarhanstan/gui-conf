import tkinter as tk
import tkinter.messagebox as tkm
import tkinter.ttk as ttk
import tkinter.filedialog as tkf
from PIL import ImageTk,Image
import polygarm as pg
import numpy as np
import os
ROOT_DIR=os.path.dirname(os.path.abspath('main.py'))

class Compare(tk.Frame):
    
    def load_file(self):
        file_name = tkf.askopenfilename(
                        filetypes=(("image files", "*.bmp;*.png;*.jpg;*.jpeg"),
                       ("All files", "*.*")))

        if file_name =="":
            return

        img=Image.open(file_name)
        img.load()
        img=img.convert("L")
        img=self.prepare_image(img)
        self.img_before=img
        self.pimage=ImageTk.PhotoImage(img)
        self.lab_bfr["image"]=self.pimage
    
    def get_var(self):
        
        if self.img_before==None:
            tkm.showwarning(title="Внимание",message="Выберите файл")
            return

        lam=self.lam_entry.get()
        try :
            self.lam=float(lam)
        except TypeError:
            tkm.showwarning(title="Внимание",message="Ожидается числовое значение параметра")
            return "Type"

        dim=self.dim_entry.get()
        if dim.isdigit():
            self.dim=int(dim)
        else:
            tkm.showwarning(title="Внимание",message="Ожидается целочисленное значение степени")
            return "Type"

        ss=self.ss_entry.get()
        if ss.isdigit():
            self.ss=int(ss)
        else:
            tkm.showwarning(title="Внимание",message="Ожидается целочисленное значение размера окна")
            return "Type"


    def push(self,fname='test'):
        print("push")

        img=self.img_before
        res=pg.Range((1-2*(self.dim%2))*pg.Rev_Lap(np.array(img,dtype=np.uint8),self.dim,self.lam))
        #print("min, max",np.min(res),np.max(res))
        self.img_res=Image.fromarray(np.uint8(res),"L")
        self.img_res.save(os.path.join(ROOT_DIR, f'{fname}.png'))
        self.img_res=ImageTk.PhotoImage(self.img_res)
        #pg.save_show(res,"test2")


        self.lab_aft["image"]=self.img_res

    def ok_button(self):
        self.get_var()
        self.push()
    
    def series(self):

        self.get_var()
        origin_lam=self.lam
        start=0
        end=1
        n=20
        pwr=1
        lam=[np.power(start+i/n,pwr)/np.power(end,pwr) for i in range(n)]
        for p in lam:
            self.lam=1+p
            self.push(fname=f'{self.dim}_{round(self.lam,6)}')
        self.lam=origin_lam
        



    def __init__(self,root):
        self.parent=root
        self.create()
    def create(self):
        self.settings_frame=tk.Frame(self.parent)
        self.settings_frame.pack(fill="x")
        #settings_frame(
        self.lam_frame=tk.LabelFrame(self.settings_frame,text="Коэф лапласа")
        self.lam_frame.pack(side="left")

        self.dim_frame=tk.LabelFrame(self.settings_frame,text="Степень")
        self.dim_frame.pack(side="left")

        self.ss_frame=tk.LabelFrame(self.settings_frame,text="Размер окна(0-full)")
        self.ss_frame.pack(side="left")

        self.dim_entry=tk.Entry(self.dim_frame,width=5)
        self.dim_entry.pack(side="left")
        self.ss_entry=tk.Entry(self.ss_frame,width=5)
        self.ss_entry.pack(side="left")

        self.lam_entry=tk.Entry(self.lam_frame,width=5)
        self.lam_entry.pack(side="left")

        self.but=tk.Button(self.settings_frame,text="OK",command=self.ok_button)
        self.but.pack(side="left")

        self.fbut=tk.Button(self.settings_frame,text="File",command=self.load_file)
        self.fbut.pack(side="left")

        self.series_but=tk.Button(self.settings_frame,text="Series",command=self.series)
        self.series_but.pack(side="left")

        self.lam_entry.insert(0,"1.01")
        self.ss_entry.insert(0,"0")
        self.dim_entry.insert(0,"1")


        #settings_frame)

        self.img_frame=tk.Frame(self.parent)
        self.img_frame.pack(fill="both")
        #img_frame(
        self.lab_bfr = tk.Label(self.img_frame)
        self.lab_bfr.pack(side="left",padx=5)
        self.lab_aft = tk.Label(self.img_frame)
        self.lab_aft.pack(side="left",padx=5)
        #img_frame)

        self.img_before: tk.PhotoImage = None



    def centerWindow(self):
        w = 290
        h = 150

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        x = (sw - w) / 2
        y = (sh - h) / 2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def prepare_image(self,img):
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        width=img.width
        height=img.height
        print("размеры получены",sw,sh,width,height)
        if width>0.4*sw or height>0.8*sh:
            if width>0.4*sw:
                kf=0.4*sw/width
                width=width*kf
                height=height*kf
            if height>0.8*sh:
                kf=0.4*sh/height
                width=width*kf
                height=height*kf
            return img.resize((round(width),round(height)))
        else:
            return img

class Series(Compare):
    def __init__(self,root):
        self.parent=root
        self.create()
    def create(self):
        self.settings_frame=tk.Frame(self.parent)
        self.settings_frame.pack(fill="x")
        #settings_frame(
        self.n_frame=tk.LabelFrame(self.settings_frame,text="from")
        self.n_frame.pack(side="left")

        self.from_frame=tk.LabelFrame(self.settings_frame,text="to")
        self.from_frame.pack(side="left")

        self.to_frame=tk.LabelFrame(self.settings_frame,text="n")
        self.to_frame.pack(side="left")

        self.power_frame=tk.LabelFrame(self.settings_frame,text="")
        self.power_frame.pack(side="left")

        self.n_entry=tk.Entry(self.n_frame,width=5)
        self.n_entry.pack(side="left")

        self.from_entry=tk.Entry(self.from_frame,width=5)
        self.from_entry.pack(side="left")
        
        self.to_entry=tk.Entry(self.to_frame,width=5)
        self.to_entry.pack(side="left")

        self.power_entry=tk.Entry(self.power_frame,width=5)
        self.power_entry.pack(side="left")

        

        self.but=tk.Button(self.settings_frame,text="OK",command=self.ok_button)
        self.but.pack(side="left")

        
        self.from_entry.insert(0,"1")
        self.to_entry.insert(0,"2")
        self.n_entry.insert(0,"10")
        self.power_entry.insert(0,'16')

        #settings_frame)

    def ok_buton(self):
        n=self.n_entry.get()
        try :
            self.n=float(n)
        except TypeError:
            tkm.showwarning(title="Внимание",message="Ожидается числовое значение")
            return "Type"

        start=self.from_entry.get()
        if start.isdigit():
            self.start=int(start)
        else:
            tkm.showwarning(title="Внимание",message="Ожидается целочисленное значение")
            return "Type"

        to=self.to_entry.get()
        if to.isdigit():
            self.to=int(to)
        else:
            tkm.showwarning(title="Внимание",message="Ожидается целочисленное значение")
            return "Type"
        
        power=self.power_entry.get()
        if power.isdigit():
            self.power=int(power)
        else:
            tkm.showwarning(title="Внимание",message="Ожидается целочисленное значение")
            return "Type"

    



root=tk.Tk()
app=Compare(root)
root.mainloop()
