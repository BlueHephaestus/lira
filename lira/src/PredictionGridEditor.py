import sys

import cv2
import numpy as np
from tkinter import *
from PIL import ImageTk, Image

from gui_base import *
from base import *

class PredictionGridEditor(object):
    def __init__(self, dataset):
        #Initialize our editor on this user's last edited image
        self.dataset = dataset

        #Ask them if they want to change the current values for resize and transparency
        if input("Would you like to change the transparency and resize factors from their current values? (These are the default values if you have not started editing) [Y\\N]: ").upper()=="Y":
            #Get resize factor and ensure it's a float 
            while True:
                self.editor_resize_factor = input("Input your Resize Factor (between 0 and 1) to scale the images by. Higher the value means higher resolution, and vice versa: ")
                if is_float(self.editor_resize_factor):
                    self.editor_resize_factor = float(self.editor_resize_factor)
                    if (self.editor_resize_factor <= 0 or 1 < self.editor_resize_factor):
                        print("ERROR: Invalid Resize Factor Input.")
                    else:
                        break
                else:
                    print("ERROR: Invalid Resize Factor Input. Exiting.")

            #Get transparency factor and ensure it's a float 
            while True:
                self.editor_transparency_factor = input("Input your Transparency Factor (between 0 and 1) to control the transparency of the colored classifications. Higher the value means higher opacity, and vice versa: ")
                if is_float(self.editor_transparency_factor):
                    self.editor_transparency_factor = float(self.editor_transparency_factor)
                    if (self.editor_transparency_factor < 0 or 1 < self.editor_transparency_factor):
                        print("ERROR: Invalid Transparency Factor Input. Exiting.")
                    else:
                        break
                else:
                    print("ERROR: Invalid Transparency Factor Input. Exiting.")
        else:
            #Just use the current values
            self.editor_resize_factor = self.dataset.progress["prediction_grids_resize_factor"]
            self.editor_transparency_factor = self.dataset.progress["prediction_grids_transparency_factor"]

        #Parameters
        self.classification_key = ["Healthy Tissue", "Type I - Caseum", "Type II", "Empty Slide", "Type III", "Type I - Rim", "Unknown/Misc."]
        self.color_key = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0), (244,66,143)]
        self.title = "LIRA Prediction Grid Editing"

        #Img + Predictions
        self.reload_img_and_predictions()

        #Window + Frame
        self.window = Tk()
        self.frame = Frame(self.window, bd=5, relief=SUNKEN)
        self.frame.grid(row=0,column=0)

        #Hard-code choice of resolution for main canvas, and hard-set scroll region as maximum shape of images
        self.main_canvas = Canvas(self.frame, bg="#000000", width=1366, height=700, scrollregion=(0,0,
                self.dataset.imgs.max_shape()[1], 
                self.dataset.imgs.max_shape()[0]))

        #Create side canvas
        self.side_canvas = Canvas(self.frame, bg="#000000", width=1366, height=68) 

        #Scrollbars
        hbar=Scrollbar(self.frame,orient=HORIZONTAL)
        hbar.pack(side=BOTTOM,fill=X)
        hbar.config(command=self.main_canvas.xview)
        vbar=Scrollbar(self.frame,orient=VERTICAL)
        vbar.pack(side=RIGHT,fill=Y)
        vbar.config(command=self.main_canvas.yview)
        self.main_canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        #Title
        self.window.title("{} - Image {}/{}".format(self.title, self.dataset.progress["prediction_grids_image"]+1, len(self.dataset.prediction_grids.before_editing)))

        #Img + Event listeners
        self.main_canvas.image = ImageTk.PhotoImage(Image.fromarray(self.img))#Literally because tkinter can't handle references properly and needs this.
        self.main_canvas_image_config = self.main_canvas.create_image(0, 0, image=self.main_canvas.image, anchor="nw")#So we can change the image later
        self.main_canvas.focus_set()
        self.main_canvas.bind("<Button 1>", self.mouse_click)
        self.main_canvas.bind("<Button 3>", self.mouse_click)
        self.main_canvas.bind("<B1-Motion>", self.mouse_move)
        self.main_canvas.bind("<B3-Motion>", self.mouse_move)
        self.main_canvas.bind("<ButtonRelease-1>", self.mouse_left_release)
        self.main_canvas.bind("<ButtonRelease-3>", self.mouse_right_release)
        self.main_canvas.bind_all("<Button-4>", self.mouse_scroll)#Scrollwheel for entire editor
        self.main_canvas.bind_all("<Button-5>", self.mouse_scroll)#Scrollwheel for entire editor
        self.main_canvas.bind("<Left>", self.left_arrow_key_press)
        self.main_canvas.bind("<Right>", self.right_arrow_key_press)
        self.main_canvas.bind("<Key>", self.key_press)
        self.main_canvas.pack(side=TOP)

        #Side Canvas
        for i, (classification, color) in enumerate(zip(self.classification_key, self.color_key)):
            #Since our colors are in BGR, and tkinter only accepts hex, we have to create a hex string for them, in RGB order.
            b, g, r = color
            hex_color_str = "#%02x%02x%02x" % (r, g, b)
            
            #We then check get the color's brightness using the relative luminance algorithm https://en.wikipedia.org/wiki/Relative_luminance
            color_brightness = (0.2126*r + 0.7152*g + 0.0722*b)/255;
            if color_brightness < 0.5:
                #Dark color, bright font.
                text_color = "white"
            else:
                #Bright color, dark font.
                text_color = "black"
            
            #Then we generate our colored label string to include the keyboard shortcut for this classification
            label_str = "Key {}: {}".format(i+1, classification)
            color_label = Label(self.side_canvas, text=label_str, bg=hex_color_str, fg=text_color, anchor="w")
            color_label.pack(fill=X)

        #Add left mouse and right mouse
        left_mouse_label = Label(self.side_canvas, text="Left Mouse: Select sections for changing classification", bg="#FFFFFF", fg="#000000", anchor="w")
        left_mouse_label.pack(fill=X)
        right_mouse_label = Label(self.side_canvas, text="Right Mouse: Select sections for full-resolution view (keep the sections small for better performance)", bg="#000000", fg="#FFFFFF", anchor="w")
        right_mouse_label.pack(fill=X)
        self.side_canvas.pack(side=BOTTOM, anchor="sw") 

        #Keeping track of which mouse button is currently held down
        self.left_mouse = False
        
        #So that if the user tries to insert classifications before they've selected any we will not do anything
        self.prediction_rect_x1 = 0
        self.prediction_rect_y1 = 0
        self.prediction_rect_x2 = 0
        self.prediction_rect_y2 = 0

        #Predictions and start
        self.window.mainloop()

    #The following functions are event handlers for our editing window. 
    def mouse_click(self, event):
        #Start a selection rect.
        #Our rectangle selections can only be made up of small rectangles of size sub_h*sub_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.

        #Get coordinates on canvas for beginning of this selection, (x1, y1)
        self.selection_x1, self.selection_y1 = get_canvas_coordinates(event)

        #Get coordinates for a rectangle outline with this point as both top-left and bot-right of the rectangle and draw it
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = get_outline_rectangle_coordinates(self.selection_x1, self.selection_y1, self.selection_x1, self.selection_y1, self.sub_h, self.sub_w)
        if event.num==1: 
            #Left Mouse Click
            self.left_click=True
            self.main_canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="darkRed", width=2, tags="classification_selection")
        else: 
            #Right Mouse Click
            self.left_click=False
            self.main_canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="darkBlue", width=2, tags="view_selection")

    def mouse_move(self, event):
        #Move the selection rect.
        #Our rectangle selections can only be made up of small rectangles of size sub_h*sub_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.

        #Get coordinates on canvas for the current end of this selection, (x2, y2)
        self.selection_x2, self.selection_y2 = get_canvas_coordinates(event)

        #Get rectangle coordinates from our initial mouse click point to this point
        rect_x1, rect_y1, rect_x2, rect_y2 = get_rectangle_coordinates(self.selection_x1, self.selection_y1, self.selection_x2, self.selection_y2)

        #Get coordinates for a new rectangle outline with this new rectangle
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, self.sub_h, self.sub_w)

        #Delete old selection rectangle and draw new one with this new rectangle outline
        if self.left_click:
            #Left Mouse Move
            self.main_canvas.delete("classification_selection")
            self.main_canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="darkRed", width=2, tags="classification_selection")
        else:
            self.main_canvas.delete("view_selection")
            self.main_canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="darkBlue", width=2, tags="view_selection")

    def mouse_left_release(self, event):
        #Set the selection rect and save its location for referencing our prediction grid.
        #Our rectangle selections can only be made up of small rectangles of size sub_h*sub_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.
        
        #Get coordinates on canvas for the end of this selection, (x2, y2)
        self.selection_x2, self.selection_y2 = get_canvas_coordinates(event)

        #Get rectangle coordinates from our initial mouse click point to this point
        rect_x1, rect_y1, rect_x2, rect_y2 = get_rectangle_coordinates(self.selection_x1, self.selection_y1, self.selection_x2, self.selection_y2)

        #Get coordinates for a new rectangle outline with this new rectangle
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, self.sub_h, self.sub_w)

        #Delete old selection rectangle and draw new finalized selection rectangle at this position
        self.main_canvas.delete("classification_selection")
        self.main_canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="red", width=2, tags="classification_selection")

        #Save the location of this outline rectangle relative to predictions so we can later update classifications in this area
        self.prediction_rect_x1 = int(outline_rect_x1/self.sub_w)
        self.prediction_rect_y1 = int(outline_rect_y1/self.sub_h)
        self.prediction_rect_x2 = int(outline_rect_x2/self.sub_w)
        self.prediction_rect_y2 = int(outline_rect_y2/self.sub_h)

    def mouse_right_release(self, event):
        #Set the selection rect and open up the selected area in a separate window at full resolution.
        #Our rectangle selections can only be made up of small rectangles of size sub_h*sub_w, so that we lock on to areas in these step sizes to allow easier rectangle selection.

        #Get coordinates on canvas for the end of this selection, (x2, y2)
        self.selection_x2, self.selection_y2 = get_canvas_coordinates(event)

        #Get rectangle coordinates from our initial mouse click point to this point
        rect_x1, rect_y1, rect_x2, rect_y2 = get_rectangle_coordinates(self.selection_x1, self.selection_y1, self.selection_x2, self.selection_y2)

        #Get coordinates for a new rectangle outline with this new rectangle
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, self.sub_h, self.sub_w)

        #Delete old selection rectangle and draw new finalized selection rectangle at this position
        self.main_canvas.delete("view_selection")
        self.main_canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="blue", width=2, tags="view_selection")

        #Open up a separate window and display the full-resolution version of the selection
        self.display_image_section(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2)
        
    def mouse_scroll(self, event):
        if event.num == 4:
            #scroll down
            self.main_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            #scroll up
            self.main_canvas.yview_scroll(1, "units")

    def left_arrow_key_press(self, event):
        #Move to the image with index i-1, unless i = 0, in which case we do nothing. AKA the previous image.

        if self.dataset.progress["prediction_grids_image"] > 0:
            #Save current predictions
            self.dataset.prediction_grids.after_editing[self.dataset.progress["prediction_grids_image"]] = self.prediction_grid

            #Change current editing image
            self.dataset.progress["prediction_grids_image"]-=1

            #Indicate Loading
            self.window.title("{} - Image {}/{} - Loading...".format(self.title, self.dataset.progress["prediction_grids_image"]+1, len(self.dataset.prediction_grids.before_editing)))
            self.window.update()

            #Reload self.img and self.prediction_grid
            self.reload_img_and_predictions()

            #Reload image displayed on canvas and predictions displayed on canvas with self.img and self.prediction_grids
            self.main_canvas.image = ImageTk.PhotoImage(Image.fromarray(self.img))#Literally because tkinter can't handle references properly and needs this.
            self.main_canvas.itemconfig(self.main_canvas_image_config, image=self.main_canvas.image)
            self.main_canvas.delete("view_selection")
            self.main_canvas.delete("classification_selection")

            #Indicate finished loading
            self.window.title("{} - Image {}/{}".format(self.title, self.dataset.progress["prediction_grids_image"]+1, len(self.dataset.prediction_grids.before_editing)))

    def right_arrow_key_press(self, event):
        #Move to the image with index i+1, unless i = img #-1, in which case we do nothing. AKA the next image.
        if self.dataset.progress["prediction_grids_image"] < len(self.dataset.imgs)-1:
            #Save current predictions
            self.dataset.prediction_grids.after_editing[self.dataset.progress["prediction_grids_image"]] = self.prediction_grid

            #Change current editing image
            self.dataset.progress["prediction_grids_image"]+=1

            #Indicate Loading
            self.window.title("{} - Image {}/{} - Loading...".format(self.title, self.dataset.progress["prediction_grids_image"]+1, len(self.dataset.prediction_grids.before_editing)))
            self.window.update()

            #Reload self.img and self.prediction_grid
            self.reload_img_and_predictions()

            #Reload image displayed on canvas and predictions displayed on canvas with self.img and self.prediction_grids
            self.main_canvas.image = ImageTk.PhotoImage(Image.fromarray(self.img))#Literally because tkinter can't handle references properly and needs this.
            self.main_canvas.itemconfig(self.main_canvas_image_config, image=self.main_canvas.image)
            self.main_canvas.delete("view_selection")
            self.main_canvas.delete("classification_selection")

            #Indicate finished loading
            self.window.title("{} - Image {}/{}".format(self.title, self.dataset.progress["prediction_grids_image"]+1, len(self.dataset.prediction_grids.before_editing)))

    def classification_key_press(self, event):
        #Change currently selected area to this classification. 
        #We update the prediction grid, but we also update the display by extracting the selected section and updating the overlay of only that section, because updating the entire image is very expensive and should be avoided.
        #First get the classification index
        i = int(event.char)-1

        #Update predictions referenced by our current classification_selection rectangle to this index and get the prediction grid section that was updated
        self.prediction_grid[self.prediction_rect_y1:self.prediction_rect_y2, self.prediction_rect_x1:self.prediction_rect_x2] = i
        self.prediction_grid_section = self.prediction_grid[self.prediction_rect_y1:self.prediction_rect_y2, self.prediction_rect_x1:self.prediction_rect_x2]

        #Save updated predictions
        self.dataset.prediction_grids.after_editing[self.dataset.progress["prediction_grids_image"]] = self.prediction_grid
        #Load the resized image section (without any overlay) referenced by our current classification_selection rectangle (no need to cast to int b/c int*int = int)
        self.img_section = self.resized_img[self.prediction_rect_y1*self.sub_h:self.prediction_rect_y2*self.sub_h, self.prediction_rect_x1*self.sub_w:self.prediction_rect_x2*self.sub_w]

        #Create new overlay on this resized image section with the prediction grid section
        self.prediction_overlay_section = np.zeros_like(self.img_section)
        for row_i, row in enumerate(self.prediction_grid_section):
            for col_i, col in enumerate(row):
                color = self.color_key[col]
                #draw rectangles of the resized sub_hxsub_w size on it
                cv2.rectangle(self.prediction_overlay_section, (col_i*self.sub_w, row_i*self.sub_h), (col_i*self.sub_w+self.sub_w, row_i*self.sub_h+self.sub_h), color, -1)

        #Combine the overlay section and the image section
        self.img_section = weighted_overlay(self.img_section, self.prediction_overlay_section, self.editor_transparency_factor)
        self.img_section = cv2.cvtColor(self.img_section, cv2.COLOR_BGR2RGB)#We need to convert so it will display the proper colors

        #Insert the now-updated image section back into the full image
        self.img[self.prediction_rect_y1*self.sub_h:self.prediction_rect_y2*self.sub_h, self.prediction_rect_x1*self.sub_w:self.prediction_rect_x2*self.sub_w] = self.img_section

        #And finally update the canvas
        self.main_canvas.image = ImageTk.PhotoImage(Image.fromarray(self.img))#Literally because tkinter can't handle references properly and needs this.
        self.main_canvas.itemconfig(self.main_canvas_image_config, image=self.main_canvas.image)

    def q_key_press(self, event):
        #(Quit) We close the editor and prompt them for if they are finished with editing or not. If they're not finished we do nothing.
        self.window.destroy()
        if input("Your prediction grid editing session has been ended. Would you like to save and continue to the next section? Once you continue, your edits can not be undone. [Y\\N]: ").upper()=='Y':
            #save this user's progress as finished editing so that we will stop the prediction grid editing phase for this user.
            self.dataset.progress["prediction_grids_finished_editing"] = True
        else:
            #Otherwise they wanna quit so quit
            sys.exit("Exiting...")

    def key_press(self, event):
        #Hub for all key press events.
        c = event.char.upper()
        if c == "Q":
            self.q_key_press(event)
        else:
            #Check if a classification key
            try:
                if (1 <= int(c) and int(c) <= len(self.classification_key)):
                    #Is a valid classification key, call handler
                    self.classification_key_press(event)
            except:
                #Not an int
                pass

        #Classification keys should remove the rectangle

    #The following functions are helper functions specific to this editor. All other GUI helpers are in the gui_base.py file.
    def reload_img_and_predictions(self):
        #Updates the self.img and self.predictions attributes. 

        #Also updates sub_h and sub_w since the prediction overlay depends on these
        self.sub_h = int(self.dataset.prediction_grids.sub_h * self.editor_resize_factor)
        self.sub_w = int(self.dataset.prediction_grids.sub_w * self.editor_resize_factor)

        self.img = self.dataset.imgs[self.dataset.progress["prediction_grids_image"]]#Load img
        self.prediction_grid = self.dataset.prediction_grids.after_editing[self.dataset.progress["prediction_grids_image"]]#Load prediction grid

        #Since our image and predictions would be slightly misalgned from each other due to rounding,
        #We compute the fx and fy img resize factors according to sub_h and sub_w to make them aligned.
        self.fy = (self.prediction_grid.shape[0]*self.sub_h)/self.img.shape[0]
        self.fx = (self.prediction_grid.shape[1]*self.sub_w)/self.img.shape[1]
        
        self.img = cv2.resize(self.img, (0,0), fx=self.fx, fy=self.fy)#Resize img
        self.resized_img = self.img#Save this so we don't have to resize later

        #Make overlay to store prediction rectangles on before overlaying on top of image
        self.prediction_overlay = np.zeros_like(self.img)

        for row_i, row in enumerate(self.prediction_grid):
            for col_i, col in enumerate(row):
                color = self.color_key[col]
                #draw rectangles of the resized sub_hxsub_w size on it
                cv2.rectangle(self.prediction_overlay, (col_i*self.sub_w, row_i*self.sub_h), (col_i*self.sub_w+self.sub_w, row_i*self.sub_h+self.sub_h), color, -1)

        self.img = weighted_overlay(self.img, self.prediction_overlay, self.editor_transparency_factor)#Overlay prediction grid onto image
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)#We need to convert so it will display the proper colors



    def display_image_section(self, x1, y1, x2, y2):
        #Given coordinates for an image section on the current resized image, get the coordinates for an image section on the full-resolution / non-resized image,
        #Then get this section on the full resolution image and display it on a new window.

        #Get updated coordinates
        x1 = int(x1/self.fx)
        y1 = int(y1/self.fy)
        x2 = int(x2/self.fx)
        y2 = int(y2/self.fy)

        #Get image section
        self.img_section = self.dataset.imgs[self.dataset.progress["prediction_grids_image"]][y1:y2, x1:x2]
        
        #Display image section on a new tkinter window
        top = Toplevel()
        top.title("Full Resolution Image Section")
        frame = Frame(top, bd=5, relief=SUNKEN)
        frame.grid(row=0,column=0)
        canvas = Canvas(frame, bg="#000000", width=self.img_section.shape[1], height=self.img_section.shape[0])
        canvas.image = ImageTk.PhotoImage(Image.fromarray(self.img_section))#Literally because tkinter can't handle references properly and needs this.
        canvas.create_image(0, 0, image=canvas.image, anchor="nw")
        canvas.pack()




