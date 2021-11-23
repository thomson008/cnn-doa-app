import math
from tkinter import *
import platform

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import UI_RESOLUTION, CHUNK


class DoaApp:
    # Window size
    WIDTH = 1000
    HEIGHT = 650

    # Canvas size
    DIM = 650

    # Distance of circle from the corner of canvas
    DIST = DIM / 12

    # Coordinates of the circle
    COORD = DIST, DIST, DIM - DIST, DIM - DIST

    # Radius of the circle
    RADIUS = DIM / 2 - DIST

    def __init__(self, top):
        self.fig, self.axs = plt.subplots(3, 2, figsize=(6, 6))
        self.lines = []

        self.prediction_running = False
        self.top = top
        self.data_frame, self.circle_frame = self.create_frames()
        self.start_button = Button(self.data_frame, text="Start", command=self.toggle_prediction,
                                   height=3, width=15, font=("Arial", 12), cursor="hand2")
        self.start_button.place(relx=0.5, y=400, anchor=CENTER)
        self.mode_selection = Button(self.data_frame, text="Change mode...", command=self.select_mode,
                                     height=2, width=15, font=("Arial", 12), cursor="hand2")
        self.mode_selection.place(relx=0.5, y=500, anchor=CENTER)

        self.exit_button = Button(self.data_frame, text="Exit", command=self.top.destroy,
                                  height=2, width=15, font=("Arial", 12), cursor="hand2")
        self.exit_button.place(relx=0.5, y=560, anchor=CENTER)
        
        if platform.system() == 'Windows':
            self.window = Toplevel(self.top)
            self.open_plot()

    def select_mode(self):
        self.data_frame.destroy()
        self.circle_frame.destroy()
        if platform.system() == 'Windows':
            self.window.destroy()

    def create_canvas(self):
        C = Canvas(self.circle_frame, bg="white", height=self.DIM, width=self.DIM)

        # Create the segmented circle in the middle of the canvas
        for i in range(0, 360, UI_RESOLUTION):
            C.create_arc(self.COORD, start=i - UI_RESOLUTION // 2, extent=UI_RESOLUTION, fill='#ebe8e8',
                         outline='#595959')

            text_R = self.RADIUS + 20
            text_x = self.DIM / 2 + text_R * math.cos(math.radians(i))
            text_y = self.DIM / 2 - text_R * math.sin(math.radians(i))

            C.create_text(text_x, text_y, fill="darkblue", font="Arial 11 bold", text=str(i))

        C.pack(fill='both')

        return C

    def create_frames(self):
        left_frame = LabelFrame(self.top, width=self.WIDTH - self.DIM - 1, height=self.HEIGHT)
        left_frame.pack(side=RIGHT)
        left_frame.pack_propagate(False)

        right_frame = LabelFrame(self.top, width=self.HEIGHT, height=self.HEIGHT)
        right_frame.pack(side=LEFT)
        right_frame.pack_propagate(False)

        return left_frame, right_frame

    def create_doa_labels(self, text, value_font_size, x, x_shift, y, y_shift):
        angle_label = Label(self.data_frame, text=text)
        angle_label.config(font=("Arial", 14), fg="#4a4a4a")
        angle_label.place(x=x, y=y)

        conf_label = Label(self.data_frame, text="Confidence")
        conf_label.config(font=("Arial", 14), fg="#4a4a4a")
        conf_label.place(x=x + x_shift, y=y)

        angle_val = Label(self.data_frame, text="-")
        angle_val.config(font=("Arial", value_font_size))
        angle_val.place(x=x, y=y + y_shift)

        conf_val = Label(self.data_frame, text="-")
        conf_val.config(font=("Arial", value_font_size))
        conf_val.place(x=x + x_shift, y=y + y_shift)

        return conf_label, conf_val, angle_label, angle_val

    def toggle_prediction(self):
        self.prediction_running = not self.prediction_running

        text = 'Stop' if self.prediction_running else 'Start'
        self.start_button.config(text=text, relief=SUNKEN if self.prediction_running else RAISED)

    def create_title_label(self):
        label = Label(self.data_frame, text="CNN DOA")
        label.config(font=("Arial", 40), fg="#4a4a4a")
        label.pack()

    def open_plot(self):
        self.window.title('Real-time signals plot')

        x = self.top.winfo_rootx()
        y = self.top.winfo_rooty()
        geom = f'+{x + self.WIDTH}+{y}'
        self.window.geometry(f'500x500{geom}')

        self.fig.suptitle('Microphone array data')
        plt.subplots_adjust(hspace=0.8, wspace=0.5)

        self.fig.suptitle('Microphone array data')
        plt.subplots_adjust(hspace=0.8, wspace=0.5)
        for i, ax in enumerate(self.axs.flat):
            ax.set_title(f'Microphone {i + 1}: {i * 60}\N{DEGREE SIGN}')
            ax.set_ylim(-300, 300)
            ax.set_xlim(0, CHUNK)

            x = np.arange(0, 2 * CHUNK, 2)
            self.lines += ax.plot(x, np.random.rand(CHUNK))

        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().pack()
