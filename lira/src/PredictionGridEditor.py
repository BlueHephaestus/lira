import sys
import cv2

from tkinter import *
from PIL import ImageTk, Image

from gui_base import *

class PredictionGridEditor(object):
    def __init__(self, dataset):
        #Initialize our editor on this user's last edited image
        self.dataset = dataset

        #Get resize factor and ensure it's a float 
        self.editor_resize_factor = input("Input your Resize Factor (between 0 and 1) to scale the images by. Higher the value means higher resolution, and vice versa: ")
        if is_float(self.editor_resize_factor):
            self.editor_resize_factor = float(self.editor_resize_factor)
            if (self.editor_resize_factor <= 0 or 1 < self.editor_resize_factor):
                sys.exit("ERROR: Invalid Resize Factor Input. Exiting.")
        else:
            sys.exit("ERROR: Invalid Resize Factor Input. Exiting.")

        #Get transparency factor and ensure it's a float 
        self.editor_transparency_factor = input("Input your Transparency Factor (between 0 and 1) to control the transparency of the colored classifications. Higher the value means higher opacity, and vice versa: ")
        if is_float(self.editor_transparency_factor):
            self.editor_transparency_factor = float(self.editor_transparency_factor)
            if (self.editor_transparency_factor < 0 or 1 < self.editor_transparency_factor):
                sys.exit("ERROR: Invalid Transparency Factor Input. Exiting.")
        else:
            sys.exit("ERROR: Invalid Transparency Factor Input. Exiting.")

        #Img + Predictions
        self.reload_img_and_detections()

        #Window + Frame
        self.window = Tk()
        self.frame = Frame(self.window, bd=5, relief=SUNKEN)
        self.frame.grid(row=0,column=0)

        #Hard-code choice of resolution for main canvas, and hard-set scroll region as maximum shape of images
        self.main_canvas = Canvas(self.frame, bg="#000000", width=1366, height=768, scrollregion=(0,0,
                self.dataset.imgs.max_shape()[1], 
                self.dataset.imgs.max_shape()[0]))

        #Create tool / key canvas
        #OI FUTURE SELF
        #Oh yea we're doing this. This is where we left off. Good luck have fun



        #Scrollbars
        hbar=Scrollbar(self.frame,orient=HORIZONTAL)
        hbar.pack(side=BOTTOM,fill=X)
        hbar.config(command=self.main_canvas.xview)
        vbar=Scrollbar(self.frame,orient=VERTICAL)
        vbar.pack(side=RIGHT,fill=Y)
        vbar.config(command=self.main_canvas.yview)
        self.main_canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        #Img + Event listeners
        self.main_canvas.image = ImageTk.PhotoImage(Image.fromarray(self.img))#Literally because tkinter can't handle references properly and needs this.
        self.main_canvas_image_config = self.main_canvas.create_image(0, 0, image=self.main_canvas.image, anchor="nw")#So we can change the image later
        self.main_canvas.focus_set()
        self.main_canvas.bind("<Button 1>", self.mouse_click)#left
        self.main_canvas.bind("<Button 3>", self.mouse_click)#right
        self.main_canvas.bind("<B1-Motion>", self.mouse_move)#left
        self.main_canvas.bind("<B3-Motion>", self.mouse_move)#right
        self.main_canvas.bind("<ButtonRelease-1>", self.mouse_left_release)
        self.main_canvas.bind("<ButtonRelease-3>", self.mouse_right_release)
        self.main_canvas.bind_all("<Button-4>", self.mouse_scroll)#Scrollwheel for entire editor
        self.main_canvas.bind_all("<Button-5>", self.mouse_scroll)#Scrollwheel for entire editor
        self.main_canvas.bind("<Left>", self.left_arrow_key_press)
        self.main_canvas.bind("<Right>", self.right_arrow_key_press)
        self.main_canvas.bind("<Key>", self.key_press)
        self.main_canvas.pack(side=LEFT)

        #Predictions and start
        self.update_predictions()
        self.window.mainloop()

    #The following functions are event handlers for our editing window. 
    def mouse_click(self, event):
        #Do the same thing for left/right click, start a selection rect.
        #Our rectangle selections can only be made up of small rectangles of size step_h*step_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.

        #Get coordinates on canvas for beginning of this selection, (x1, y1)
        self.selection_x1, self.selection_y1 = get_canvas_coordinates(event)

        #Get coordinates for a rectangle outline with this point as both top-left and bot-right of the rectangle and draw it
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = get_outline_rectangle_coordinates(self.selection_x1, self.selection_y1, self.selection_x1, self.selection_y1, self.step_h, self.step_w)
        self.canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="darkRed", width=2, tags="selection")

    def mouse_move(self, event):
        #Do the same thing for left/right move, move the selection rect.
        #Our rectangle selections can only be made up of small rectangles of size step_h*step_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.

        #Get coordinates on canvas for the current end of this selection, (x2, y2)
        self.selection_x2, self.selection_y2 = get_canvas_coordinates(event)

        #Get rectangle coordinates from our initial mouse click point to this point
        rect_x1, rect_y1, rect_x2, rect_y2 = get_rectangle_coordinates(self.selection_x1, self.selection_y1, self.selection_x2, self.selection_y2)

        #Get coordinates for a new rectangle outline with this new rectangle
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, self.step_h, self.step_w)

        #Delete old selection rectangle and draw new one with this new rectangle outline
        self.canvas.delete("selection")
        self.canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="darkRed", width=2, tags="selection")

    def mouse_left_release(self, event):
        #Remove all detections from the area in this rectangle selection, then add detections of size rect_hxrect_w in intervals step_hxstep_w in the area in this rectangle selection.
        #   Then update the detections.
        #Our rectangle selections can only be made up of small rectangles of size step_hxstep_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.
        
        #Get coordinates on canvas for the end of this selection, (x2, y2)
        self.selection_x2, self.selection_y2 = get_canvas_coordinates(event)

        #Get rectangle coordinates from our initial mouse click point to this point
        rect_x1, rect_y1, rect_x2, rect_y2 = get_rectangle_coordinates(self.selection_x1, self.selection_y1, self.selection_x2, self.selection_y2)

        #Get coordinates for a new rectangle outline with this new rectangle
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, self.step_h, self.step_w)

        #Delete old selection rectangle along with all detection rectangles in this new selection rectangle.
        self.canvas.delete("selection")
        
        for i, detection in reversed(list(enumerate(self.detections))):
            if detection_in_rect(detection, [outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2], self.rect_h, self.rect_w):
                del self.detections[i]
                
        #add detections of size rect_hxrect_w in intervals step_hxstep_w in the area in this rectangle selection.
        for x1 in range(int(outline_rect_x1), int(outline_rect_x2-self.step_w)+1, self.step_w):
            for y1 in range(int(outline_rect_y1), int(outline_rect_y2-self.step_h)+1, self.step_h):
                detection = [x1, y1, x1+self.rect_w, y1+self.rect_h]
                self.detections.append(detection)

        #Finally update all detections
        self.update_detections()

    def mouse_right_release(self, event):
        #Remove all detections from the area in this rectangle selection, Then update the detections.
        #Our rectangle selections can only be made up of small rectangles of size step_h*step_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.

        #Get coordinates on canvas for the end of this selection, (x2, y2)
        self.selection_x2, self.selection_y2 = get_canvas_coordinates(event)

        #Get rectangle coordinates from our initial mouse click point to this point
        rect_x1, rect_y1, rect_x2, rect_y2 = get_rectangle_coordinates(self.selection_x1, self.selection_y1, self.selection_x2, self.selection_y2)

        #Get coordinates for a new rectangle outline with this new rectangle
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, self.step_h, self.step_w)

        #Delete old selection rectangle along with all detection rectangles in this new selection rectangle.
        self.canvas.delete("selection")
        
        for i, detection in reversed(list(enumerate(self.detections))):
            if detection_in_rect(detection, [outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2], self.rect_h, self.rect_w):
                del self.detections[i]

        #Finally update all detections
        self.update_detections()

    def mouse_scroll(self, event):
        if event.num == 4:
            #scroll down
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            #scroll up
            self.canvas.yview_scroll(1, "units")

    def left_arrow_key_press(self, event):

    def right_arrow_key_press(self, event):

    def q_key_press(self, event):

    def key_press(self, event):

    #The following functions are helper functions specific to this editor. All other GUI helpers are in the gui_base.py file.
    def update_detections(self):
        #Put all our detections for the current image as rectangles on the current canvas image. 
        self.canvas.delete("detection")
        for detection in self.detections:
            self.canvas.create_rectangle(detection[0], detection[1], detection[2], detection[3], fill='', outline="red", width=2, tags="detection")

        #Convert our local detections copy back into a np array, scale it back up, cast it to int, and update the stored detections.
        self.dataset.type_one_detections.after_editing[self.dataset.progress["type_ones_image"]] = (np.array(self.detections)/self.editor_resize_factor).astype(int)

    def reload_img_and_detections(self):
        #Updates the self.img and self.detections attributes. 
        self.img = cv2.resize(self.dataset.imgs[self.dataset.progress["type_ones_image"]], (0,0), 
                fx=self.editor_resize_factor,
                fy=self.editor_resize_factor)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)#We need to convert so it will display the proper colors
        self.detections = list(self.dataset.type_one_detections.after_editing[self.dataset.progress["type_ones_image"]]*self.editor_resize_factor)#Make list so we can append


