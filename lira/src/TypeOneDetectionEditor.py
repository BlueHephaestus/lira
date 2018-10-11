import sys
import cv2

from tkinter import *
from PIL import ImageTk, Image

from gui_base import *

class TypeOneDetectionEditor(object):
    def __init__(self, dataset):
        #Initialize our editor on this user's last edited image
        self.dataset = dataset

        #Editor Parameters
        self.editor_resize_factor = 0.1#amount to resize images for display
        self.rect_h = 64#Height of each rectangle when displayed
        self.rect_w = 64#Width of each rectangle when displayed
        self.step_h = int(self.rect_h/2.)#Height of each step when selecting rectangles
        self.step_w = int(self.rect_w/2.)#Width of each step when selecting rectangles

        #Img + Detections
        self.reload_img_and_detections()

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
        self.canvas_image_config = self.canvas.create_image(0, 0, image=self.canvas.image, anchor="nw")#So we can change the image later
        self.canvas.focus_set()
        self.canvas.bind("<Button 1>", self.mouse_click)#left
        self.canvas.bind("<Button 3>", self.mouse_click)#right
        self.canvas.bind("<B1-Motion>", self.mouse_move)#left
        self.canvas.bind("<B3-Motion>", self.mouse_move)#right
        self.canvas.bind("<ButtonRelease-1>", self.mouse_left_release)
        self.canvas.bind("<ButtonRelease-3>", self.mouse_right_release)
        self.canvas.bind_all("<Button-4>", self.mouse_scroll)#Scrollwheel for entire editor
        self.canvas.bind_all("<Button-5>", self.mouse_scroll)#Scrollwheel for entire editor
        self.canvas.bind("<Left>", self.left_arrow_key_press)
        self.canvas.bind("<Right>", self.right_arrow_key_press)
        self.canvas.bind("<Key>", self.key_press)
        self.canvas.pack(side=LEFT)

        #Detections and start
        self.update_detections()
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
        #Move to the image with index i-1, unless i = 0, in which case we do nothing. AKA the previous image.

        if self.dataset.progress["type_ones_image"] > 0:
            #Change current editing image
            self.dataset.progress["type_ones_image"]-=1

            #Reload self.img and self.detections
            self.reload_img_and_detections()

            #Reload image displayed on canvas and detections displayed on canvas with self.img and self.detections
            self.canvas.image = ImageTk.PhotoImage(Image.fromarray(self.img))#Literally because tkinter can't handle references properly and needs this.
            self.canvas.itemconfig(self.canvas_image_config, image=self.canvas.image)
            self.canvas.delete("selection")
            self.canvas.delete("detection")
            self.update_detections()

    def right_arrow_key_press(self, event):
        #Move to the image with index i+1, unless i = img #-1, in which case we do nothing. AKA the next image.
        if self.dataset.progress["type_ones_image"] < len(self.dataset.imgs)-1:
            #Change current editing image
            self.dataset.progress["type_ones_image"]+=1

            #Reload self.img and self.detections
            self.reload_img_and_detections()

            #Reload image displayed on canvas and detections displayed on canvas with self.img and self.detections
            self.canvas.image = ImageTk.PhotoImage(Image.fromarray(self.img))#Literally because tkinter can't handle references properly and needs this.
            self.canvas.itemconfig(self.canvas_image_config, image=self.canvas.image)
            self.canvas.delete("selection")
            self.canvas.delete("detection")
            self.update_detections()

    def q_key_press(self, event):
        #(Quit) We close the editor and prompt them for if they are finished with editing or not. If they're not finished we do nothing.
        self.window.destroy()
        if input("Your type one detection editing session has been ended. Would you like to continue? Once you continue, your edits can not be undone. [Y\\N]: ").upper()=='Y':
            #save this user's progress as finished editing so that we will stop the type one detection editing phase for this user.
            self.dataset.progress["type_ones_finished_editing"] = True
        else:
            #Otherwise they wanna quit so quit
            sys.exit("Exiting...")

    def key_press(self, event):
        #Hub for all key press events.
        c = event.char.upper()
        if c == "Q":
            self.q_key_press(event)

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


