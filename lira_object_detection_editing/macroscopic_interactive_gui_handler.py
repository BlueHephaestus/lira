"""
Main functions for displaying the GUI for our Object Detection Editor,
    further documentation found in each function.

-Blake Edwards / Dark Element
"""

import Tkinter as tk
from Tkinter import *

import PIL
from PIL import Image, ImageTk

import cv2

import numpy as np

class InteractiveGUI(object):

    """
    Our class for an interactive gui for LIRA Live.
    """
    def __init__(self, dual_monitor):
        """
        Arguments:
            dual_monitor: Boolean for if we are using two monitors or not. 
                Shrinks the width of our display if we are, and leaves normal if not.

        Returns:
            Initialises all of our necessary values and methods for our interactive GUI. 
                There is too much to include here, view individual documentation below.
        """
        """
        Our main image, and main rects that we are currently working with. 
            remember rects must stay a list so we can easily change it's size.
        """
        self.np_img = np.array([-1])
        self.rects = [-1]

        """
        We also have 
            rect_h and rect_w: The size of our individual detected rectangles in our image,
            and step_h and step_w: The size of our step size or stride length between each rectangle in our image.
                (the minimal distance possible between two rects)
            These have to also be changed each image to account for each image having a different resize factor.
        """
        self.rect_h = -1
        self.rect_w = -1
        self.step_h = -1
        self.step_w = -1

        """
        Indicator / Flag value(s) for the status of the session, 
            so that our main file can know when to get the next image.
        """
        self.flag_next = False

        """
        Variables to store the location where we opened our bulk select rectangle
        """
        self.bulk_select_initial_x = 0
        self.bulk_select_initial_y = 0

        """
        For configuring the window, you may definitely have to mess with these percentages as I had to mess with them a lot to get reasonable ones for our monitors.
            I may also figure out a more general way to do this in the future.
        """
        self.screen_height_padding_percentage = 0.20
        if dual_monitor:
            self.main_canvas_percentage = 1.0
            self.screen_width_padding_percentage = 0.51
        else:
            self.main_canvas_percentage = 0.85
            self.screen_width_padding_percentage = 0.02

    def get_relative_coordinates(self, event):
        """
        Arguments:
            event: A tkinter event
        Returns:
            (x, y): the x and y coordinates of our event on our parent canvas
        """
        canvas = event.widget
        return canvas.canvasx(event.x), canvas.canvasy(event.y)

    def get_rectangle_coordinates(self, x1, y1, x2, y2):
        """
        Arguments:
            (x1, y1), (x2, y2): Two sets of coordinates defining a rectangle between the two sets.
        
        Returns:
            Two new sets of coordinates for the top-left and bottom-right corner of the rectangle defined by the input coordinates.

            We do this by simply handling each possible orientation of the points relative to each other,
                and returning the correct combination of the points.
        """
        if x1 <= x2 and y1 <= y2:
            """
            1
             \
              2
            """
            return x1, y1, x2, y2

        elif x1 > x2 and y1 <= y2:
            """
              1
             /
            2 
            """
            return x2, y1, x1, y2
        elif x1 <= x2 and y1 > y2:
            """
              2
             /
            1 
            """
            return x1, y2, x2, y1

        elif x1 > x2 and y1 > y2:
            """
            2
             \
              1
            """
            return x2, y2, x1, y1

    def get_outline_rectangle_coordinates(self, rect_x1, rect_y1, rect_x2, rect_y2, step_h, step_w):
        """
        Arguments:
            rect_x1, rect_y1, rect_x2, rect_y2: Two pairs of coordinates for the top left and bottom right corners of a rectangle
            step_h, step_w: The size of our step, could be subsections, rectangles, or whatever - just use this to determine intervals for our outline rect.
                This can also be thought of as the stride length as in convolutional networks.

        Returns:
            Two new pairs of coordinates for a new rectangle, 
                which outlines all the subsections the original rectangle touches.

        Luckily, this is easily done with a simple modular arithmetic formula I made up.
        """
        outline_rect_x1 = np.floor(rect_x1/self.step_w)*self.step_w 
        outline_rect_y1 = np.floor(rect_y1/self.step_h)*self.step_h
        outline_rect_x2 = np.ceil(rect_x2/self.step_w)*self.step_w
        outline_rect_y2 = np.ceil(rect_y2/self.step_h)*self.step_h

        return outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2

    def rect_in_outline_rect(self, rect, outline_rect):
        """
        Arguments:
            rect: Rect we are checking, to see if it is inside the outline_rect region.
                Should be of format [x1, y1, x2, y2]
            outline_rect: Rect we are checking our rect against, should be >= the area of our rect. 
                Should be of format [x1, y1, x2, y2]

        Returns:
            True if our rect is inside our outline_rect region,
            False if not.
        """
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = outline_rect
        rect_x1, rect_y1, rect_x2, rect_y2 = rect

        outline_rect_x2 -= self.rect_w
        outline_rect_y2 -= self.rect_h
        """
        If it's in the bounds, return true. We modified our x2, y2 cords of our outline_rect in order to make it so that
            we only had to worry about the x1, y1 cords of our rect when checking. That way, we could treat the rect as a point,
            and then check if it was inside or on the boundary of our rectangle specified by our NEW outline_rect cords.
        So with our modification it became just 4 conditionals in an if statement.
        """
        return ((outline_rect_x1 <= rect_x1 and rect_x1 <= outline_rect_x2) and (outline_rect_y1 <= rect_y1 and rect_y1 <= outline_rect_y2))

    def redraw_rects(self, canvas):
        """
        Arguments:
            canvas: Canvas object to redraw rectangles on.
        Returns:
            I use canvas as an argument instead of event because this is NOT an event handler. 
            This is called manually when we need to erase all existing rects from the canvas, 
                and then draw all rects in our self.rects list to replace them.
            So when it's done, the canvas will be updated to match our rects list, if it wasn't already.
        """
        #Erase all detected / added rects
        canvas.delete("detected_rect")

        #Loop through and draw all rects in our self.rects list to replace them.
        for rect in self.rects:
            x1, y1, x2, y2 = rect
            #print x2 - x1, y2 - y1
            canvas.create_rectangle(x1, y1, x2, y2, fill='', outline="red", width=2, tags="detected_rect")

        #That's all folks
        return

    def mouse_click(self, event):
        """
        Handler for when the left or right click is pressed
        """
        """
        Arguments:
            event: The event that triggered this function.

        Returns:
            Starts our bulk select rectangle for the user to select multiple classifications at once,
                by setting the initial x and y coordinates of our bulk select rectangle.
            Since we've just pressed down the button, we store the relative coordinates into our class variables 
                to use in computing where to display the rectangle.
            Though our clicks have different functions:
                Left click = add
                Right click = remove
            These are not used until the mouse is released, so we can call this function regardless of the mouse button used.
        """
        print "Mouse clicked, opening bulk select rectangle..."
        canvas = event.widget
        self.bulk_select_initial_x, self.bulk_select_initial_y = self.get_relative_coordinates(event)

        """
        Since it is more user-friendly to immediately open the rectangle even if the user hasn't moved their mouse yet,
            since they may want to only select one rectangle, and so they might not move it,
            we draw the bulk select rectangle immediately.

        We use our initial x and y to get new coordinates that outline all the subsections that our bulk select rectangle encompasses.
            This way, the user can see what rectangles will be selected when they release the mouse.
        """
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = self.get_outline_rectangle_coordinates(self.bulk_select_initial_x, self.bulk_select_initial_y, self.bulk_select_initial_x, self.bulk_select_initial_y, self.step_h, self.step_w)

        """
        Then we draw our new outline rectangle.
        """
        canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="darkRed", width=2, tags="bulk_select_rect")

    def mouse_move(self, event):
        """
        Handler for when the left or right click is moved while pressed
        """
        """
        Arguments:
            event: The event that triggered this function.

        Returns:
            Clicking opens a bulk select rectangle, and moving should update the position of that bulk select rectangle.
            Since we will be performing either an add or remove operation (depending on the button clicked) after the mouse is released
                (because it's faster and also because I like the intuitive idea behind it)
                we will not have the bulk select rectangle move freely - rather it will lock onto the individual selection rectangles as it passes over them.
            I also like this because the user can easily see what they are selecting, rather than being unsure what rectangles they are selecting.
        """
        rel_x, rel_y = self.get_relative_coordinates(event)

        canvas = event.widget
        """
        Then, we update our rectangle to be drawn from our initial x and y coordinates to our current x and y coordinates.
        Our create_rectangle method requires the top left point's coordinates, then the bottom right's coordinates.
            Since we don't always have these points, we just have two points that are diagonally opposite, 
                we compute the necessary coordinates with our get_rectangle_coordinates() function.
            Then, we delete any existing rectangles, and
            we create a new rectangle with a red outline and a tag so we can delete it whenever we need to.
        """
        rect_x1, rect_y1, rect_x2, rect_y2 = self.get_rectangle_coordinates(self.bulk_select_initial_x, self.bulk_select_initial_y, rel_x, rel_y)

        """
        Using these, we get new coordinates that outline all the subsections that our bulk select rectangle encompasses.
            This way, the user can see what rectangles will be selected when they release the mouse.
        """
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = self.get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, self.step_h, self.step_w)

        """
        Then we delete our last bulk select rectangle, now that we can draw a new one to replace it.
        """
        canvas.delete("bulk_select_rect")

        """
        And we draw our new outline rectangle.
        """
        canvas.create_rectangle(outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2, fill='', outline="darkRed", width=2, tags="bulk_select_rect")

    def mouse_left_release(self, event):
        """
        Handler for when the left click is released.
        """
        """
        Arguments:
            event: The event that triggered this function.

        Returns:
            Since this is for the left mouse, we will add rectangles to the entire area inside of our bulk select rectangle.
            To do this, it will remove all existing rectangles in the area, then go through and add new ones throughout the entire area.
        """
        print "Left Mouse released, adding rectangles to bulk selection area..."
        canvas = event.widget

        """
        First, we get the outline_rect coordinates for the last time, the same way we got them in our mouse_move function. 
        Since it's the same process as that function, i'm not repeating the documentation. 
        """
        rel_x, rel_y = self.get_relative_coordinates(event)
        rect_x1, rect_y1, rect_x2, rect_y2 = self.get_rectangle_coordinates(self.bulk_select_initial_x, self.bulk_select_initial_y, rel_x, rel_y)
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = self.get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, self.step_h, self.step_w)
        canvas.delete("bulk_select_rect")
        
        """
        Since if we drew a rectangle, it would be immediately deleted after this function is finished, 
            we save the trouble and don't draw a new rectangle.
        Then, we do our magic. 
        In order to remove the rectangles, we could sort based on x1 and y1, however this wouldn't save much time since we're dealing with clusters. 
        So we use our outline_rect coordinates to check each rectangle in our main list to see if it lies inside, and if so remove it.
        """
        outline_rect = [outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2]

        """
        Since we are looping through our rects, and we could remove the rect we are currently testing if a given condition is true, 
            we have to loop in reverse order so that we don't accidentally skip indices.
        Since we are looping in reverse order, we also keep track of the index in reverse order. 
            We need the index so we can still delete elements.
        """
        i = len(self.rects)-1
        for rect in reversed(self.rects):
            #Check if our rect is inside the outline rect
            if self.rect_in_outline_rect(rect, outline_rect):
                #If so, delete.
                del self.rects[i]
            #Decrement index since we're going right -> left
            i-=1
        
        """
        Alright that was easy, now we just fill our outline_rect region with rects of shape (self.rect_h, self.rect_)
        We do this by stepping self.step_w and self.step_h in our region's boundaries to get the x1 and y1, then add self.rect_w and self.rect_h
            to x1 and y1 respectively to get x2 and y2.
            So that x2 = x1 + self.rect_w and y2 = y1 + self.rect_h
            And our rects are always of size rect_h x rect_w.
        Note: we do outline_rect_x2 - self.step_w since we add self.rect_w to our final coordinate pair, and don't want to go outside our bounds when adding new rects.
            Same for outline_rect_y2
        We then just append them to our main self.rects list.
        """
        for x1 in range(int(outline_rect_x1), int(outline_rect_x2-self.step_w), self.step_w):
            for y1 in range(int(outline_rect_y1), int(outline_rect_y2-self.step_h), self.step_h):
                rect = [x1, y1, x1+self.rect_w, y1+self.rect_h]
                self.rects.append(rect)
        """
        Unfortunately, the above 2 loops don't add any rects if either 
            int(outline_rect_x2 - self.step_w) == int(outline_rect_x1) 
                or 
            int(outline_rect_y2 - self.step_h) == int(outline_rect_y1)
            are True.
        The first case is for when our width is step_w, but still < rect_w, 
        and the second is for when our height is step_h, but still < rect_h.

        AKA the first case is a thin and tall rectangle, and the second is a short and fat rectangle.

        If we only draw a rectangle of size step_h x step_w (if we just click the mouse once),
            then both these cases will be true. 

        That is the only time both will be true though, otherwise we only have one of the above cases true.

        So there are now 3 different if statements we need:
            1. Case #1 is true
            2. Case #2 is true
            3. Case #1 and Case #2 are true

        Because of the way the loops in the first two if statements are structured,
            If we have both Case #1 and Case #2 as true we wouldn't add any rectangle
                in the first and second if statements.
            Because of this, we need a third for that one last edge case where we have just clicked the mouse
                and only have a tiny rect of size step_h x step_w
        """
        if int(outline_rect_x2 - self.step_w) == int(outline_rect_x1):
            #thin and tall rectangle, loop through height and add
            x1 = int(outline_rect_x1)
            for y1 in range(int(outline_rect_y1), int(outline_rect_y2-self.step_h), self.step_h):
                rect = [x1, y1, x1+self.rect_w, y1+self.rect_h]
                self.rects.append(rect)

        if int(outline_rect_y2 - self.step_h) == int(outline_rect_y1):
            #short and fat rectangle, loop through width and add
            y1 = int(outline_rect_y1)
            for x1 in range(int(outline_rect_x1), int(outline_rect_x2-self.step_w), self.step_w):
                rect = [x1, y1, x1+self.rect_w, y1+self.rect_h]
                self.rects.append(rect)

        if int(outline_rect_x2 - self.step_w) == int(outline_rect_x1) and int(outline_rect_y2 - self.step_h) == int(outline_rect_y1):
            #rectangle of smallest possible size, step_h x step_w. So we just add one rectangle here.
            x1 = int(outline_rect_x1)
            y1 = int(outline_rect_y1)
            rect = [x1, y1, x1+self.rect_w, y1+self.rect_h]
            self.rects.append(rect)

        """
        Since we've updated our self.rects list, we need to update the rects on our canvas
            to match, so we redraw the rectangles on that. 
        """
        self.redraw_rects(canvas)

        """
        With that we are done, the entire region has had new rectangles added to it.
        """
        return

    def mouse_right_release(self, event):
        """
        Handler for when the right click is released.
        """
        """
        Arguments:
            event: The event that triggered this function.

        Returns:
            Since this is for the right mouse, we will remove rectangles from the entire area inside of our bulk select rectangle.
            NOTE: This function is almost completely identical to the mouse_left_release function up to the part where that function adds rectangles, 
                since our function doesn't add in any new rectangles.
        """
        print "Right Mouse released, adding rectangles to bulk selection area..."
        canvas = event.widget

        """
        First, we get the outline_rect coordinates for the last time, the same way we got them in our mouse_move function. 
        Since it's the same process as that function, i'm not repeating the documentation. 
        """
        rel_x, rel_y = self.get_relative_coordinates(event)
        rect_x1, rect_y1, rect_x2, rect_y2 = self.get_rectangle_coordinates(self.bulk_select_initial_x, self.bulk_select_initial_y, rel_x, rel_y)
        outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2 = self.get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, self.step_h, self.step_w)
        canvas.delete("bulk_select_rect")
        
        """
        Since if we drew a rectangle, it would be immediately deleted after this function is finished, 
            we save the trouble and don't draw a new rectangle.
        Then, we do our magic. 
        In order to remove the rectangles, we could sort based on x1 and y1, however this wouldn't save much time since we're dealing with clusters. 
        So we use our outline_rect coordinates to check each rectangle in our main list to see if it lies inside, and if so remove it.
        """
        outline_rect = [outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2]

        """
        Since we are looping through our rects, and we could remove the rect we are currently testing if a given condition is true, 
            we have to loop in reverse order so that we don't accidentally skip indices.
        Since we are looping in reverse order, we also keep track of the index in reverse order. 
            We need the index so we can still delete elements.
        """
        i = len(self.rects)-1
        for rect in reversed(self.rects):
            #Check if our rect is inside the outline rect
            if self.rect_in_outline_rect(rect, outline_rect):
                #If so, delete.
                del self.rects[i]
            #Decrement index since we're going right -> left
            i-=1

        """
        Since we've updated our self.rects list, we need to update the rects on our canvas
            to match, so we redraw the rectangles on that. 
        """
        self.redraw_rects(canvas)
        
        """
        With that we are done, the entire region has had any existing rectangles removed.
        """
        return

    def key_press(self, event):
        """
        Handler for when a key is pressed. 
        """
        """
        Arguments:
            event: The event that triggered this function.

        Returns:
            Depending on the key pressed, we may either:
                1. Call another event handler if we call a command shortcut, or
                2. We may do nothing if we have not selected any valid keyboard shortcut
            We only have a next image event handler here, so that's all we check for.
        """
        canvas = event.widget
        c = event.char
        print "%s pressed..." % c
        c = c.upper()
        if c == 'N':
            """
            Our next keybind, call the associated function handler.
            """
            self.next_img()
        """
        Whatever happens, we are done with this handler now.
        """
        return

    def next_img(self):
        """
        Function for going to the next image, called from our next image keyboard shortcut handler.
        """
        """
        Arguments:
            (none)

        Returns:
            We destroy our window and change flag,
                so that our main function (where our InteractiveGUI object has been instantiated)
                knows to give us a new image or quit if this is the last image, 
                and also get the updated rects list from our InteractiveGUI object.
        """
        print "Going to next image..."
        self.window.destroy()
        self.flag_next = True

    @staticmethod
    def get_relative_canvas_dimensions(screen_width, screen_height, main_canvas_percentage, screen_width_padding_percentage, screen_height_padding_percentage, relative_dim):
        """
        Arguments:
            screen_width, screen_height: The width and height of our screen
            main_canvas_percentage, tool_canvas_percentage: The percentages to give to each of our canvases. 
                Should both be >= 0 and <= 1 and sum to <= 1, if you want to retain sanity.
            screen_width_padding_percentage, screen_height_padding_percentage: Percentage of screen_width / screen_height to pad in each direction.
            relative_dim: "w" or "h", Dimension to split up into portions so we can fit our main_canvas and tool_canvas in it. 
                e.g. if "h", we stack them on top of each other, if "w", we put them side-to-side

        Returns:
            We pad our screen w and h the necessary padding amount,
            Get the height and widths of our canvases wrt our relative dim, and return these.
        """
        """
        First, pad both the necessary screen padding amount.
        """
        screen_width = screen_width - screen_width_padding_percentage*screen_width
        screen_height = screen_height - screen_height_padding_percentage*screen_height

        """
        Then invert this percentage so we don't give the opposite of what we want in our upcoming calculations
        """
        main_canvas_percentage = 1-main_canvas_percentage

        """
        Then we give each section it's corresponding percentage of the padded relative dim 
        """
        if relative_dim == "w":
            """
            Compute the relative width 
            """
            main_canvas_width = screen_width - main_canvas_percentage*screen_width

            """
            And assign the full height
            """
            main_canvas_height = screen_height

        elif relative_dim == "h":
            """
            Compute the height
            """
            main_canvas_height = screen_height - main_canvas_percentage*screen_height

            """
            And assign the full width
            """
            main_canvas_width = screen_width

        else:
            sys.exit("Incorrect relative_dim parameter input for get_relative_canvas_dimensions()")

        return main_canvas_width, main_canvas_height

    def start_interactive_session(self):
        """
        Arguments:
            (none)

        Returns:
            Starts all our window objects and initialises all necessary values and methods for our interactive GUI.
                There is too much to include here, view individual documentation below.
        """
        """
        Check if we have an image to display and rects to update and rect_h and rect_w, otherwise exit.
            Even if our image doesn't initially have any rects, the list will be set to [], not [-1] as is the default.
            This way we know we haven't set our rects yet if the list == [-1]
        """
        if np.all(self.np_img==-1):
            sys.exit("Error: You must assign an image to display before starting an interactive GUI session!")

        if np.all(self.rects == -1):
            sys.exit("Error: You must assign rectangles to update before starting an interactive GUI session!")

        if (self.rect_h == -1 or self.rect_w == -1):
            sys.exit("Error: You must assign the individual height and width of your individual rectangles (rect_h and rect_w) before starting an interactive GUI session!")

        if (self.step_h == -1 or self.step_w == -1):
            sys.exit("Error: You must assign the height and width of your step size / stride length (step_h and step_w) before starting an interactive GUI session!")

        """
        Initialize our main Tkinter window object, and use that to get our screen width and height.
            Initializing our window like this doesn't open any new windows, it just initializes the object.
            We would use the window we used for the screen height and width in our __init__, however due to how Tkinter
                works with classes I couldn't get it to then display images in the canvas of the window when I initialized
                the window in a different class function. So we initialize it here.
        """
        self.window = Tk()

        """
        Get screen height and width
        """
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        """
        Using our screen width and measurements for main canvas and padding,
            assign our main canvas to be this size with padding and dual monitor factors factored in.
        We do this here so that we can use our canvas width and height attributes where our session has been instantiated.
        """
        self.main_canvas_width, self.main_canvas_height = self.get_relative_canvas_dimensions(
                self.screen_width, 
                self.screen_height, 
                self.main_canvas_percentage,
                self.screen_width_padding_percentage,
                self.screen_height_padding_percentage,
                relative_dim="w")

        """
        First convert our np array from BGR to RGB
        """
        self.np_img = cv2.cvtColor(self.np_img, cv2.COLOR_BGR2RGB)

        """
        Convert our image from numpy array, to PIL image, then to Tkinter image.
            Numpy -> PIL -> Tkinter
        """
        self.img = ImageTk.PhotoImage(Image.fromarray(self.np_img))

        """
        Initialize our main frame
        """
        frame = Frame(self.window, bd=5, relief=SUNKEN)
        frame.grid(row=0,column=0)

        """
        Initialize our main canvas with the majority of our screen, and the scroll region as the size of our image,
        """
        main_canvas=Canvas(frame, bg='#000000', width=self.main_canvas_width, height=self.main_canvas_height, scrollregion=(0,0,self.np_img.shape[1],self.np_img.shape[0]))

        """
        Initialize horizontal and vertical scrollbars on main canvas
        """
        hbar=Scrollbar(frame,orient=HORIZONTAL)
        hbar.pack(side=BOTTOM,fill=X)
        hbar.config(command=main_canvas.xview)
        vbar=Scrollbar(frame,orient=VERTICAL)
        vbar.pack(side=RIGHT,fill=Y)
        vbar.config(command=main_canvas.yview)
        main_canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        """
        Add our image to the main canvas, in the upper left corner.
        """
        main_canvas.create_image(0, 0, image=self.img, anchor="nw")

        """
        Add our event listeners to the main canvas
        """
        main_canvas.focus_set()

        """
        Handler for when the left or right click is pressed.
            These both call the same event handler since we do the same regardless of the button pressed.
        """
        main_canvas.bind("<Button 1>", self.mouse_click)#left
        main_canvas.bind("<Button 3>", self.mouse_click)#right

        """
        Handler for when the left or right click is moved while pressed
            These both call the same event handler since we do the same regardless of the button moved while pressed.
        """
        main_canvas.bind("<B1-Motion>", self.mouse_move)#left
        main_canvas.bind("<B3-Motion>", self.mouse_move)#right

        """
        Handler for when the left click is released
        """
        main_canvas.bind("<ButtonRelease-1>", self.mouse_left_release)

        """
        Handler for when the right click is released
        """
        main_canvas.bind("<ButtonRelease-3>", self.mouse_right_release)

        """
        Handler for when any key is pressed
        """
        main_canvas.bind("<Key>", self.key_press)

        """
        Open our canvas in the main window.
        """
        main_canvas.pack(side=LEFT)

        """
        Draw the rectangles in our self.rects list onto the canvas.
        """
        self.redraw_rects(main_canvas)

        """
        Initialize main window and main loop.
        """
        self.window.mainloop()

