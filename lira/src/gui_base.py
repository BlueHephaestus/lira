import numpy as np

def get_canvas_coordinates(event):
    #Get coordinates on canvas for this selection (event.x, event.y)
    return event.widget.canvasx(event.x), event.widget.canvasy(event.y)

def get_rectangle_coordinates(x1, y1, x2, y2):
    #Get the top-left and bot-right points for the rectangle defined by the two points.
    #We do this by handling the 4 different orientations possible.
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

def get_outline_rectangle_coordinates(rect_x1, rect_y1, rect_x2, rect_y2, step_h, step_w):
    #Get a new rectangle made entirely of smaller rectangles of size step_hxstep_w which outlines the area encompassed by the given rectangle.
    #Luckily, this is easily done with a simple modular arithmetic formula I came up with.
    outline_rect_x1 = np.floor(rect_x1/step_w)*step_w 
    outline_rect_y1 = np.floor(rect_y1/step_h)*step_h
    outline_rect_x2 = np.ceil(rect_x2/step_w)*step_w
    outline_rect_y2 = np.ceil(rect_y2/step_h)*step_h

    return outline_rect_x1, outline_rect_y1, outline_rect_x2, outline_rect_y2

def detection_in_rect(detection, rect, rect_h, rect_w):
    #Check if the given detection is inside our rect.
    detection_x1, detection_y1, detection_x2, detection_y2 = detection
    rect_x1, rect_y1, rect_x2, rect_y2 = rect

    #So that we only check if a rect's top-left corner is inside our outline rect, not the entire rect. I've found this to be more intuitive.
    rect_x2 -= rect_w
    rect_y2 -= rect_h

    return ((rect_x1 <= detection_x1 and detection_x1 <= rect_x2) and (rect_y1 <= detection_y1 and detection_y1 <= rect_y2))
