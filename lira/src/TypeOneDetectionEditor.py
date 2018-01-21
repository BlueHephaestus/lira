import cv2

from tkinter import *
from PIL import ImageTk, Image

from gui_base import *

class TypeOneDetectionEditor(object):
    def __init__(self, dataset):
        #Initialize our editor on this user's last edited image

        #Editor Parameters
        self.editor_resize_factor = 0.1#amount to resize images for display
        self.rect_h = 64#Height of each rectangle when displayed
        self.rect_w = 64#Width of each rectangle when displayed
        self.step_h = 32#Height of each step when selecting rectangles
        self.step_w = 32#Width of each step when selecting rectangles

        #Img + Detections
        self.dataset = dataset

        self.img = cv2.resize(self.dataset.imgs[self.dataset.progress["type_ones_image"]], (0,0), 
                fx=self.editor_resize_factor,
                fy=self.editor_resize_factor)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)#We need to convert so it will display the proper colors
        self.detections = self.dataset.type_one_detections.before_editing[self.dataset.progress["type_ones_image"]]*self.editor_resize_factor

        #Window + Frame
        self.window = Tk()
        self.frame = Frame(self.window, bd=5, relief=SUNKEN)
        self.frame.grid(row=0,column=0)

        #Hard-code choice of resolution for canvas and scroll region as maximum shape of images*resize_factor
        self.canvas = Canvas(self.frame, bg="#000000", width=1366, height=768, scrollregion=(0,0,
                self.dataset.imgs.max_shape()[1]*self.editor_resize_factor, 
                self.dataset.imgs.max_shape()[0]*self.editor_resize_factor))

        #Scrollbars
        hbar=Scrollbar(self.frame,orient=HORIZONTAL)
        hbar.pack(side=BOTTOM,fill=X)
        hbar.config(command=self.canvas.xview)
        vbar=Scrollbar(self.frame,orient=VERTICAL)
        vbar.pack(side=RIGHT,fill=Y)
        vbar.config(command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        #Img + Event listeners
        self.canvas.image = ImageTk.PhotoImage(Image.fromarray(self.img))#Literally because tkinter can't handle references properly and needs this.
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor="nw")
        self.canvas.focus_set()
        self.canvas.bind("<Button 1>", self.mouse_click)#left
        self.canvas.bind("<Button 3>", self.mouse_click)#right
        self.canvas.bind("<B1-Motion>", self.mouse_move)#left
        self.canvas.bind("<B3-Motion>", self.mouse_move)#right
        self.canvas.bind("<ButtonRelease-1>", self.mouse_left_release)
        self.canvas.bind("<ButtonRelease-3>", self.mouse_right_release)
        self.canvas.bind("<Key>", self.key_press)
        self.canvas.pack(side=LEFT)

        #Detections and start
        self.render_detections()
        self.window.mainloop()

    #The following functions are event handlers for our editing window. 
    def mouse_click(self, event):
        #Do the same thing for left/right click, start a selection rect.
        #Our rectangle selections can only be made up of small rectangles of size step_h*step_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.

        #Get coordinates on canvas for beginning of this selection, (x1, y1)
        self.selection_x1, self.selection_y1 = get_canvas_coordinates(event)

        #Get coordinates for a rectangle outline with this point as both top-left and bot-right of the rectangle and draw it
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = get_outline_rectangle_coordinates(self.selection_x1, self.selection_y1, self.selection_x1, self.selection_y1, self.step_h, self.step_w)
        canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="darkRed", width=2, tags="bulk_select_rect")

    def mouse_move(self, event):
        #Do the same thing for left/right move, move the selection rect.
        #Our rectangle selections can only be made up of small rectangles of size step_h*step_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.
        pass

    def mouse_left_release(self, event):
        #Remove all detections from the area in this rectangle selection, then add detections of size rect_hxrect_w in intervals step_hxstep_w in the area in this rectangle selection.
        #   Then re-render the detections.
        #Our rectangle selections can only be made up of small rectangles of size step_hxstep_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.
        pass

    def mouse_right_release(self, event):
        #Remove all detections from the area in this rectangle selection, Then re-render the detections.
        #Our rectangle selections can only be made up of small rectangles of size step_h*step_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.
        pass

    def key_press(self, event):
        #Either q - quit, left arrow - previous image, or right arrow - next image.
        pass

    #The following functions are helper functions specific to this editor. All other GUI helpers are in the gui_base.py file.
    def render_detections(self):
        #Put all our detections for the current image as rectangles on the current canvas image. 
        self.canvas.delete("detection")
        for detection in self.detections:
            print(detection)
            self.canvas.create_rectangle(detection[0], detection[1], detection[2], detection[3], fill='', outline="red", width=2, tags="detection")

