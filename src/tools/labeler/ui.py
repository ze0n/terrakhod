from tkinter import *
from tkinter.ttk import *

#window = Tk()
#window.mainloop()


class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master=master
        window = master
        pad=0
        self._geom='200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
        master.bind('<Escape>',self.toggle_geom)

        window.title("Welcome to LikeGeeks app")
        Label(window, text="Settings").grid(row=0)

        Label(window, text="Databases").grid(row=2)
        listboxDatabases = Listbox(window)
        listboxDatabases.grid(row=3)

        for item in ["one", "two", "three", "four"]:
            listboxDatabases.insert(END, item)

        Label(window, text="Datasets").grid(row=4)
        listboxDatasets = Listbox(window)
        listboxDatasets.grid(row=5)

        for item in ["one", "two", "three", "four"]:
            listboxDatasets.insert(END, item)

        Separator(orient="vertical").grid(column=1)



    def toggle_geom(self,event):
        geom=self.master.winfo_geometry()
        print(geom,self._geom)
        self.master.geometry(self._geom)
        self._geom=geom


class Fullscreen_Window:

    def createWindow(self, window):
        window.title("Welcome to LikeGeeks app")
        Label(window, text="Settings").grid(row=0)

        window.configure(background="Gray")

        Label(window, text="Databases").grid(row=2)
        listboxDatabases = Listbox(window)
        listboxDatabases.grid(row=3)

        for item in ["one", "two", "three", "four"]:
            listboxDatabases.insert(END, item)

        Label(window, text="Dataset").grid(row=4)
        listboxDatasets = Listbox(window)
        listboxDatasets.grid(row=5)

        for item in ["one", "two", "three", "four"]:
            listboxDatasets.insert(END, item)

        Separator(orient="vertical").grid(column=1)

        FMas = Frame(window)
        FMas.grid(column=2, row=0, sticky=(N,E,S,W))

        FMas.grid_rowconfigure(1, weight=1)
        FMas.grid_columnconfigure(2, weight=1)

        L1 = Label(FMas, text="Frame 1 Contents")
        L1.grid(row=0, column=0)

        Can1 = Canvas(FMas, bg="Yellow")
        Can1.grid(row=1, column=0, sticky=(N,W))


    def __init__(self):
        self.tk = Tk()
        #self.tk.attributes('-zoomed', True)  # This just maximizes it so we can see the window. It's nothing to do with fullscreen.
        self.tk.attributes("-fullscreen", True)
        
        self.createWindow(self.tk)

        self.frame = Frame(self.tk).grid(column=2)
        #self.frame.pack()

        self.state = False
        self.tk.bind("<F11>", self.toggle_fullscreen)
        self.tk.bind("<Escape>", self.end_fullscreen)

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.tk.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.tk.attributes("-fullscreen", False)
        return "break"

if __name__ == '__main__':
    w = Fullscreen_Window()
    w.tk.mainloop()

#window=Tk()
#app=FullScreenApp(window)
#window.mainloop()