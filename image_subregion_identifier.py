import Tkinter
import tkFileDialog
from PIL import ImageTk
import PIL.Image
import os
import cv2
import numpy as np

from isi_lib import utils

BACKGROUND_COLOR = '#ededed'

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 720

PREVIEW_SIZE = 200  # height & width of preview in pixels

PAD_SMALL = 2
PAD_MEDIUM = 4
PAD_LARGE = 8
PAD_EXTRA_LARGE = 14


class Application(Tkinter.Frame):

    def __init__(self, master):

        Tkinter.Frame.__init__(self, master=master)

        self.image_name = None
        self.image_dir = None

        # Detected regions will be saved as a dictionary with the bounding
        # rectangles canvas ID as the key. The value will be another dictionary
        # containing the contour itself and the rectangle coordinates
        self.regions = None

        self.region_class = Tkinter.StringVar()

        self.master.minsize(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self.master.title("Image Sub-region Identifier")

        self.main_frame = Tkinter.Frame(self.master, bg=BACKGROUND_COLOR)
        self.main_frame.pack(
            fill=Tkinter.BOTH,
            expand=True,
            padx=0,
            pady=0
        )

        self.left_frame = Tkinter.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        self.left_frame.pack(
            fill=Tkinter.BOTH,
            expand=True,
            side=Tkinter.LEFT,
            padx=0,
            pady=0
        )

        self.right_frame = Tkinter.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        self.right_frame.pack(
            fill=Tkinter.Y,
            expand=False,
            side=Tkinter.LEFT,
            padx=PAD_MEDIUM,
            pady=(40, 10)
        )

        file_chooser_frame = Tkinter.Frame(self.left_frame, bg=BACKGROUND_COLOR)
        file_chooser_frame.pack(
            fill=Tkinter.X,
            expand=False,
            anchor=Tkinter.N,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        file_chooser_button = Tkinter.Button(
            file_chooser_frame,
            text='Choose Image File...',
            command=self.choose_files
        )
        file_chooser_button.pack(side=Tkinter.LEFT)

        # the canvas frame's contents will use grid b/c of the double
        # scrollbar (they don't look right using pack), but the canvas itself
        # will be packed in its frame
        canvas_frame = Tkinter.Frame(self.left_frame, bg=BACKGROUND_COLOR)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.pack(
            fill=Tkinter.BOTH,
            expand=True,
            anchor=Tkinter.N,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        self.canvas = Tkinter.Canvas(canvas_frame, cursor="cross")

        self.scrollbar_v = Tkinter.Scrollbar(
            canvas_frame,
            orient=Tkinter.VERTICAL
        )
        self.scrollbar_h = Tkinter.Scrollbar(
            canvas_frame,
            orient=Tkinter.HORIZONTAL
        )
        self.scrollbar_v.config(command=self.canvas.yview)
        self.scrollbar_h.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.scrollbar_v.set)
        self.canvas.config(xscrollcommand=self.scrollbar_h.set)

        self.canvas.grid(
            row=0,
            column=0,
            sticky=Tkinter.N + Tkinter.S + Tkinter.E + Tkinter.W
        )
        self.scrollbar_v.grid(row=0, column=1, sticky=Tkinter.N + Tkinter.S)
        self.scrollbar_h.grid(row=1, column=0, sticky=Tkinter.E + Tkinter.W)

        region_buttons_frame = Tkinter.Frame(
            self.right_frame,
            bg=BACKGROUND_COLOR
        )
        region_buttons_frame.pack(
            fill=Tkinter.BOTH,
            expand=False,
            anchor=Tkinter.N,
            pady=PAD_MEDIUM
        )

        identify_region_button = Tkinter.Button(
            region_buttons_frame,
            text='Identify Region',
            command=self.identify_region
        )
        identify_region_button.pack(side=Tkinter.LEFT, anchor=Tkinter.N)

        # frame displaying prediction for target sub-region
        stats_frame = Tkinter.Frame(
            self.right_frame,
            bg=BACKGROUND_COLOR,
            highlightthickness=1,
            highlightbackground='gray'
        )
        stats_frame.pack(
            fill=Tkinter.BOTH,
            expand=False,
            anchor=Tkinter.N,
            pady=PAD_LARGE,
            padx=PAD_MEDIUM
        )
        region_class_frame = Tkinter.Frame(
            stats_frame,
            bg=BACKGROUND_COLOR
        )
        region_class_frame.pack(
            fill=Tkinter.BOTH,
            expand=True,
            anchor=Tkinter.N,
            pady=PAD_SMALL,
            padx=PAD_SMALL
        )
        region_class_desc_label = Tkinter.Label(
            region_class_frame,
            text="Region class: ",
            bg=BACKGROUND_COLOR
        )
        region_class_desc_label.pack(side=Tkinter.LEFT, anchor=Tkinter.N)
        region_class_label = Tkinter.Label(
            region_class_frame,
            textvariable=self.region_class,
            bg=BACKGROUND_COLOR
        )
        region_class_label.pack(side=Tkinter.RIGHT, anchor=Tkinter.N)

        # preview frame holding small full-size depiction of chosen image
        preview_frame = Tkinter.Frame(
            self.right_frame,
            bg=BACKGROUND_COLOR,
            highlightthickness=1,
            highlightbackground='black'
        )
        preview_frame.pack(
            fill=Tkinter.NONE,
            expand=False,
            anchor=Tkinter.S,
            side=Tkinter.BOTTOM
        )

        self.preview_canvas = Tkinter.Canvas(
            preview_frame,
            highlightthickness=0
        )
        self.preview_canvas.config(width=PREVIEW_SIZE, height=PREVIEW_SIZE)
        self.preview_canvas.pack(anchor=Tkinter.S, side=Tkinter.BOTTOM)

        # setup some button and key bindings
        self.canvas.bind("<ButtonPress-1>", self.on_draw_button_press)
        self.canvas.bind("<B1-Motion>", self.on_draw_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_draw_release)

        self.canvas.bind("<ButtonPress-2>", self.on_pan_button_press)
        self.canvas.bind("<B2-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_button_release)

        self.canvas.bind("<Configure>", self.canvas_size_changed)

        self.scrollbar_h.bind("<B1-Motion>", self.update_preview)
        self.scrollbar_h.bind("<ButtonRelease-1>", self.update_preview)
        self.scrollbar_v.bind("<B1-Motion>", self.update_preview)
        self.scrollbar_v.bind("<ButtonRelease-1>", self.update_preview)

        self.preview_canvas.bind("<ButtonPress-1>", self.move_preview_rectangle)
        self.preview_canvas.bind("<B1-Motion>", self.move_preview_rectangle)

        self.rect = None

        self.start_x = None
        self.start_y = None

        self.pan_start_x = None
        self.pan_start_y = None

        self.image = None
        self.tk_image = None
        self.preview_image = None
        self.preview_rectangle = None

        self.pack()

    def on_draw_button_press(self, event):
        # starting coordinates
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create a new rectangle if we don't already have one
        if self.rect is None:
            self.rect = self.canvas.create_rectangle(
                self.start_x,
                self.start_y,
                self.start_x,
                self.start_y,
                outline='#00ff00',
                width=2
            )

    def on_draw_move(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        # update rectangle size with mouse position
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    # noinspection PyUnusedLocal
    def on_draw_release(self, event):
        if self.rect is None or self.image is None:
            return

        corners = self.canvas.coords(self.rect)
        corners = tuple([int(c) for c in corners])
        region = self.image.crop(corners)

        if 0 in region.size:
            # either height or width is zero, do nothing
            return

        target = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2HSV)

    def on_pan_button_press(self, event):
        self.canvas.config(cursor='fleur')

        # starting position for panning
        self.pan_start_x = int(self.canvas.canvasx(event.x))
        self.pan_start_y = int(self.canvas.canvasy(event.y))

    def pan_image(self, event):
        self.canvas.scan_dragto(
            event.x - self.pan_start_x,
            event.y - self.pan_start_y,
            gain=1
        )
        self.update_preview(None)

    # noinspection PyUnusedLocal
    def on_pan_button_release(self, event):
        self.canvas.config(cursor='cross')

    def identify_region(self):
        if self.rect is None or self.image is None:
            return

        corners = self.canvas.coords(self.rect)
        corners = tuple([int(c) for c in corners])
        region = self.image.crop(corners)

        # identify best class for region
        predicted_class = utils.predict(region, self.image_name)

        self.region_class.set(predicted_class)

    def set_preview_rectangle(self):
        x1, x2 = self.scrollbar_h.get()
        y1, y2 = self.scrollbar_v.get()

        self.preview_rectangle = self.preview_canvas.create_rectangle(
            int(x1 * PREVIEW_SIZE) + 1,
            int(y1 * PREVIEW_SIZE) + 1,
            int(x2 * PREVIEW_SIZE),
            int(y2 * PREVIEW_SIZE),
            outline='#00ff00',
            width=2,
            tag='preview_rect'
        )

    # noinspection PyUnusedLocal
    def update_preview(self, event):
        if self.preview_rectangle is None:
            # do nothing
            return

        x1, x2 = self.scrollbar_h.get()
        y1, y2 = self.scrollbar_v.get()

        # current rectangle position
        rx1, ry1, rx2, ry2 = self.preview_canvas.coords(
            self.preview_rectangle
        )

        delta_x = int(x1 * PREVIEW_SIZE) + 1 - rx1
        delta_y = int(y1 * PREVIEW_SIZE) + 1 - ry1

        self.preview_canvas.move(
            self.preview_rectangle,
            delta_x,
            delta_y
        )

    def move_preview_rectangle(self, event):
        if self.preview_rectangle is None:
            # do nothing
            return

        x1, y1, x2, y2 = self.preview_canvas.coords(self.preview_rectangle)

        half_width = float(x2 - x1) / 2
        half_height = float(y2 - y1) / 2

        if event.x + half_width >= PREVIEW_SIZE - 1:
            new_x = PREVIEW_SIZE - (half_width * 2) - 1
        else:
            new_x = event.x - half_width

        if event.y + half_height >= PREVIEW_SIZE - 1:
            new_y = PREVIEW_SIZE - (half_height * 2) - 1
        else:
            new_y = event.y - half_height

        self.canvas.xview(
            Tkinter.MOVETO,
            float(new_x) / PREVIEW_SIZE
        )

        self.canvas.yview(
            Tkinter.MOVETO,
            float(new_y) / PREVIEW_SIZE
        )

        self.update()
        self.update_preview(None)

    # noinspection PyUnusedLocal
    def canvas_size_changed(self, event):
        self.preview_canvas.delete('preview_rect')
        self.set_preview_rectangle()

    def choose_files(self):
        selected_file = tkFileDialog.askopenfile('r')

        if selected_file is None:
            # do nothing, user cancelled file dialog
            return

        self.canvas.delete('all')
        self.rect = None
        self.region_class.set('')

        # some of the files may be 3-channel 16-bit/chan TIFFs, which
        # PIL doesn't support. OpenCV can read these, but converts them
        # to 8-bit/chan. So, we'll open all images in OpenCV first,
        # then create a PIL Image to finally create an ImageTk PhotoImage
        cv_img = cv2.imread(selected_file.name)

        self.image = PIL.Image.fromarray(
            cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB),
            'RGB'
        )
        height, width = self.image.size
        self.canvas.config(scrollregion=(0, 0, height, width))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=Tkinter.NW, image=self.tk_image)

        # have to force an update of the UI else the canvas scroll bars
        # will not have updated fast enough to get their positions for
        # drawing the preview rectangle
        self.update()

        tmp_preview_image = self.image.resize(
            (PREVIEW_SIZE, PREVIEW_SIZE),
            PIL.Image.ANTIALIAS
        )
        self.preview_canvas.delete('all')
        self.preview_image = ImageTk.PhotoImage(tmp_preview_image)
        self.preview_canvas.create_image(
            0,
            0,
            anchor=Tkinter.NW,
            image=self.preview_image
        )
        self.set_preview_rectangle()

        self.image_name = os.path.basename(selected_file.name)
        self.image_dir = os.path.dirname(selected_file.name)

root = Tkinter.Tk()
app = Application(root)
root.mainloop()
