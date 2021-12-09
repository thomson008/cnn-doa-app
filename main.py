import sys
from tkinter import *

from multi_source_app import MultiSourceApp
from single_source_app import SingleSourceApp


class MainApp:
    # Window size
    WIDTH = 1000
    HEIGHT = 650

    def __init__(self):
        self.top = Tk()
        self.top.title('ML DoA recognition')
        self.top.geometry(f'{self.WIDTH}x{self.HEIGHT}')
        self.top.resizable(False, False)
        label = Label(self.top, text="Choose mode:")
        label.config(font=("Arial", 20), fg="#4a4a4a")
        label.place(relx=0.5, rely=0.35, anchor=CENTER)

        self.single_source_button = Button(self.top, text="Single source", command=self.start_single_source,
                                           height=3, width=15, font=("Arial", 12), cursor="hand2")
        self.single_source_button.place(relx=0.4, rely=0.5, anchor=CENTER)
        self.multi_source_button = Button(self.top, text="Multi source", command=self.start_multi_source,
                                          height=3, width=15, font=("Arial", 12), cursor="hand2")
        self.multi_source_button.place(relx=0.6, rely=0.5, anchor=CENTER)

    def start_single_source(self):
        single_source_app = SingleSourceApp(self.top)
        single_source_app.run()

    def start_multi_source(self):
        multi_source_app = MultiSourceApp(self.top)
        multi_source_app.run()


if __name__ == '__main__':
    try:
        app = MainApp()
        app.top.mainloop()
    except Exception:
        sys.exit()
