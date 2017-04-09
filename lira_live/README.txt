Opening a session:
  1. Open a command prompt in this directory. If you are on windows, this can be done with Ctrl+Shift+RightClick -> Open command prompt...
  2. Input the command `python lira_live1.py` (without the ``s) to open a new interactive session.

Commands for using LIRA-Live in an interactive session:
  1. Left Mouse Click:
    Left clicking will allow you to select subsections for classification, 
      you can also select multiple subsections with the same method as dragging the mouse allows you to do with desktop icons.

  2. Right Mouse Click:
    Right clicking and moving will allow you to move the viewport / area you are currently viewing around by manipulating the scroll bars.

  3. "Alpha Transparency" Slider:
    Changing the value of this parameter affects the transparency of the overlayed colored rectangles (predictions).
    0 = full transparency, rectangles/predictions not visible.
    1 = no transparency, rectangles/predictions completely opaque and lesions not visible.
    If you change this, the program will remember it, even if you quit the session. Default is 0.33 .

  4. "Zoom Percentage" Slider:
    Percentages are 20%, 100%, and 200%
    Changes the zoom just like a web browser.
    Same behavior as previous slider.

  5. Number keys 1/2/3/.../N
    Notated on the right of the window, you can see the numerical keys for classifications. 
    Pressing one of these will change the classification of the currently selected subsection(s).

  6. Refresh Session button / R key
    This button will refresh the displayed image, to show changes in classifications or parameters, such as the Alpha transparency.
    The changes made to either classifications or parameters do happen regardless of if refresh is pressed, it is merely a tool to show the user updates.

  7. Next Image button / N key
    This button will go to the next image, saving the new predictions and parameters.

  8. Quit Session / Q key
    This button will end the session, saving the new predictions and parameters for all images up to, BUT NOT INCLUDING, the currently open image.
    This is true for every image but the very last one, at which point it will include the currently open image because there are no others.

Information on saved parameters:
  The position where the last user left off with LIRA-live, as well as the parameter values, 
    can be found in the interactive_session_metadata.pkl file. 
  Do not modify them.

  They are saved when quitting a session.

Information on quitting / ending a session / saving data:
  Do not quit by closing the window, this may result in you losing updated parameter(s) or valuable progress.
    As long as you exit in a normal way, using my commands, you will not lose any progress. However, if you do exit by closing the window, you may lose some progress.
  If you wish to quit, note that any changes to predictions will be saved, however depending on how you quit will affect where they are saved.
    For example, if you quit on the second image:
      The first image and predictions will be saved in the location for training data, to be used later for the LIRA network.
      The first image's predictions will also be updated automatically in the place where all the current predictions (not used for training) are stored.
      The second image will not save anything in the location for training data,
      However the second image's predictions will also be updated in the same manner.
    This will be true for every scenario except for if the image is the final image in the entire dataset.

Feel free to email/text/call me if you have any questions, or want anything changed for your ease of use!
