#importing libraries to be used
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import subprocess
import sys

# creating a screenshot directory
SCREENSHOT_DIR = "Screenshot"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# defining folder properties
# width ratio, height ratio, margin, color, and text color
FOLDER_W_RATIO = 0.15  
FOLDER_H_RATIO = 0.15  
FOLDER_MARGIN = 12   
FOLDER_COLOR = (235, 221, 164)
FOLDER_ICON_TEXT_COLOR = (0, 0, 0)

#opening the folder of screenshots
def open_folder(path):
    path = os.path.abspath(path)
    subprocess.run(["open", path])

#mouse event handler to check if there's a click that occurs within the folder's box
folder_rect = None 
mouse_down = False

#I wasn't sure on how to code up this part, so I used ChatGPT to help 
def on_mouse(event, x, y, flags, param):
    # tracking the mouse's state
    global mouse_down
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
        if folder_rect is not None:
            # checking if the mouse's coordinates are within the rectangle defined for folder
            x1, y1, x2, y2 = folder_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                # open screenshots folder
                open_folder(SCREENSHOT_DIR)

# initiating mediapipe -- defining the hands objects and some characteristics
# maximum number of hands, the minimum detection & tracking confidence 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# helper module inside mediapipe that allows mediapipe to draw landmarks and connecting lines
mp_draw = mp.solutions.drawing_utils

# creating the camera frame and video loop
cap = cv2.VideoCapture(0)
cv2.namedWindow("Drawing", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Drawing", on_mouse)

# initializing some variables including previous coordinates, initial color and canvas
prev_x, prev_y = None, None
draw_enabled = True
touching_middle = False
blue, green, red = 0, 0, 255
canvas = None

# forever loop
while True:
    # start the camera frame and start reading from the camera stream
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    # defining the shape of the frame (the width and height of the frame)
    h, w, _ = frame.shape

    # if the canvas doesn't exist or the canvas shape does not match the current frame size, create a new canvas
    if canvas is None or canvas.shape[0] != h or canvas.shape[1] != w:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # converts camera frame colors to the right format and run mediapipe hands on it 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # drawing the lines that connect the landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            LM = mp_hands.HandLandmark
            # a dictionary of all the hand landmarks that are relevant and their relevant ids
            ids = {
                "index_tip": LM.INDEX_FINGER_TIP,
                "index_pip": LM.INDEX_FINGER_PIP,
                "thumb_tip": LM.THUMB_TIP,
                "pinky_tip": LM.PINKY_TIP,
                "middle_tip": LM.MIDDLE_FINGER_TIP,
                "ring_tip": LM.RING_FINGER_TIP
            }

            # creating a dictionary of the name of the landmarks and the corresponding mediapipe landmark objects
            lms = {name: handLms.landmark[idx] for name, idx in ids.items()}

            # getting the pixel coordinates for each landmark from the normalized coordinates that mediapipe provides
            def xy(lm):
                return int(lm.x * w), int(lm.y * h)

            # obtaining the coordinates for each landmark we're interested in (fingertips)
            ix, iy = xy(lms["index_tip"])
            px, py = xy(lms["index_pip"])
            tx, ty = xy(lms["thumb_tip"])
            pinky_x, pinky_y = xy(lms["pinky_tip"])
            middle_x, middle_y = xy(lms["middle_tip"])
            ring_x, ring_y = xy(lms["ring_tip"])

            # "clamping" the coordinate so it doesn't go out of bounds
            # without these two lines, my code easily broke when my hand moved too fast
            sx = max(0, min(ix, w - 1))
            sy = max(0, min(iy - 30, h - 1))
            # obtaining the color of the pixel that my pointer finger is pointing to
            b, g, r = frame[sy, sx]

            # drawing using pointer finger if drawing is enabled
            if iy < py:
                cv2.circle(frame, (ix, iy), 10, (int(b), int(g), int(r)), cv2.FILLED)
                if prev_x is not None and draw_enabled:
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (blue, green, red), 5)
                prev_x, prev_y = ix, iy
            else:
                prev_x, prev_y = None, None

            # changing/picking color if my pointer finger is touching my thumb
            if np.hypot(ix - tx, iy - ty) < 50:
                blue, green, red = int(b), int(g), int(r)

            # clearing the canvas if the thumb an pinky are touching
            if np.hypot(pinky_x - tx, pinky_y - ty) < 50:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # screenshot (if ring fingertip is touching thumb)
            if np.hypot(ring_x - tx, ring_y - ty) < 50:
                filename = os.path.join(SCREENSHOT_DIR, f"screenshot_{int(time.time())}.png")
                # save the composed final frame (frame + drawing)
                # I used AI to help me with this part
                gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
                drawing_only = cv2.bitwise_and(canvas, canvas, mask=mask)
                final_for_save = cv2.add(frame_bg, drawing_only)
                cv2.imwrite(filename, final_for_save)
                print(f"[Saved] {filename}")

            # turn the drawing feature on/off (by touching middle fingertip to thumb)
            dist_middle_thumb = np.hypot(middle_x - tx, middle_y - ty)
            # switching value of the boolean expression that describes if drawing is enabled
            if dist_middle_thumb < 50 and not touching_middle:
                draw_enabled = not draw_enabled
                touching_middle = True
            elif dist_middle_thumb >= 50:
                touching_middle = False

    # composing the final image to display
    # creating a mask so as to display the drawing in full color and clearly without distorting the coloration of the background 
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    drawing_only = cv2.bitwise_and(canvas, canvas, mask=mask)
    final = cv2.add(frame_bg, drawing_only)

    # creating a folder icon at the specified location
    fw = int(w * FOLDER_W_RATIO)
    fh = int(h * FOLDER_H_RATIO)
    x2 = w - FOLDER_MARGIN
    y2 = FOLDER_MARGIN + fh
    x1 = x2 - fw
    y1 = FOLDER_MARGIN

    # update global rect for mouse handler
    folder_rect = (x1, y1, x2, y2)

    # adding text to the bottom of the screen
    cv2.putText(
        final, 
        "Thumb to Index: Change Color; Thumb to Middle: Drawer On/Off; Thumb to Ring Finger: Screenshot; Thumb to Pinky: Clear Frame", 
        (20, h-20), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6,              
        (0, 255, 0),      
        1,               
        cv2.LINE_AA
    )

    # used AI for this part
    # draw main folder body
    color = FOLDER_COLOR 
    cv2.rectangle(final, (x1, y1 + int(fh*0.25)), (x2, y2), color, cv2.FILLED)
    # draw folder tab
    tab_w = int(fw * 0.45)
    tab_h = int(fh * 0.25)
    cv2.rectangle(final, (x1 + 6, y1), (x1 + 6 + tab_w, y1 + tab_h), color, cv2.FILLED)
    # outline
    cv2.rectangle(final, (x1, y1), (x2, y2), (0,0,0), 2)
    # description of the folder
    cv2.putText(final, "Photos Folder", (x1 + 8, y1 + fh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FOLDER_ICON_TEXT_COLOR, 1, cv2.LINE_AA)

    # show camera frame and drawing
    cv2.imshow("Drawing", final)
    
    # adding and updating variable "folder_rect" (I used AI for this line)
    globals()['folder_rect'] = folder_rect

    # escaping the camera frame
    if cv2.waitKey(1) & 0xFF == 27:
        break

#closing the windows
cap.release()
cv2.destroyAllWindows()

