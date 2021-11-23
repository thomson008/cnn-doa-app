from doa_app import DoaApp
from tkinter import *

from multi_source_predictor import MultiSourcePredictor
from utils import UI_RESOLUTION, AZIMUTH_RESOLUTION


class MultiSourceApp(DoaApp):
    def color_arcs(self, C, predictions):
        arcs = []
        for i, conf in enumerate(predictions):
            R = self.RADIUS * conf ** 0.5
            x_1 = y_1 = self.DIM / 2 - R

            coord = x_1, y_1, x_1 + 2 * R, y_1 + 2 * R

            if conf > 0.5:
                fill = '#78ebb3'
                outline = '#2ca86b'
                arc = C.create_arc(coord, start=i * UI_RESOLUTION - 5, extent=UI_RESOLUTION, fill=fill, outline=outline,
                                   width=2)
                arcs.append(arc)

        self.top.update()

        # Delete all arcs to draw them again at next iteration
        for arc in arcs:
            C.delete(arc)

    def create_title_label(self):
        super().create_title_label()
        label = Label(self.data_frame, text="Multi source")
        label.config(font=("Arial", 14), fg="#4a4a4a")
        label.pack()

    def create_labels(self):
        self.create_title_label()

        x = 40
        x_shift = 140
        value_font_size = 25

        y_shift = 30
        y_first_azimuth = 120
        az_conf_label_1, az_conf_val_1, azimuth_label_1, azimuth_val_1 = self.create_doa_labels(
            'Azimuth 1', value_font_size, x, x_shift, y_first_azimuth, y_shift)

        y_second_azimuth = 250
        az_conf_label_2, az_conf_val_2, azimuth_label_2, azimuth_val_2 = self.create_doa_labels(
            'Azimuth 2', value_font_size, x, x_shift, y_second_azimuth, y_shift)

        return azimuth_label_1, azimuth_val_1, az_conf_label_1, az_conf_val_1, \
            azimuth_label_2, azimuth_val_2, az_conf_label_1, az_conf_val_2

    def run(self):
        C = self.create_canvas()
        az_label, az_val, az_conf_label, az_conf_val, el_label, el_val, el_conf_label, el_conf_val = self.create_labels()
        predictor = MultiSourcePredictor(self.lines, self.fig)

        while True:
            predictor.is_active = self.prediction_running
            predictions = predictor.az_current_predictions

            # Color arcs based on model probabilities
            try:
                self.color_arcs(C, predictions)
                angles = [(angle * AZIMUTH_RESOLUTION, conf) for angle, conf in enumerate(predictions) if conf > 0.5]
                if len(angles) == 1:
                    az_val.config(text=f'{angles[0][0]}\N{DEGREE SIGN}')
                    az_conf_val.config(text=f'{round(angles[0][1] * 100, 1)}%')
                elif len(angles) == 2:
                    el_val.config(text=f'{angles[1][0]}\N{DEGREE SIGN}')
                    el_conf_val.config(text=f'{round(angles[1][1] * 100, 1)}%')
                else:
                    az_val.config(text='-')
                    az_conf_val.config(text='-')
                    el_val.config(text='-')
                    el_conf_val.config(text='-')
            except TclError:
                predictor.is_active = False
                return
