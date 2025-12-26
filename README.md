! CURRENTLY REQUIRES MACOS SYSTEM TO RUN WEBCAM-SPECIFIC FUNCTIONS

DEMO:

![Image](https://github.com/user-attachments/assets/0d261196-ea3b-4a23-ade1-71e36a8860de)

This is a gesture-controlled virtual drawing app using computer vision that allows the user to draw directly in their camera frame with built-in functionalities. These functionalities include allowing the user to draw with their pointer finger, select colors, toggle the pen (to switch drawing mode on or off), clear the canvas and take screenshots. This is a project primarily built off of OpenCV and MediaPipe Hands, with additional libraries such as NumPy and Time: these libraries provide the program the ability to recognize hand landmarks and process frames in real time.

The program can be run on VSCode or any Python compiler. In particular, a working webcam is needed, as is a macOS system (for the command used to access and open the webcam). The Python libraries which are needed include OpenCV, MediaPipe and NumPy, which can be downloaded from their respective website’s release page.

Once the program starts running, a webcam window titled “Drawing” will appear (this may take up to 10 seconds as the environment is set up). A frame will then appear, displaying the camera frame’s contents, with instructions labeled in green at the bottom of the frame and a blue folder icon for screenshots on the top right corner. After the frame is set up, the user may begin drawing.

The program currently supports several drawing-related features as follows. 

- Drawing: The user can draw by tracing their index finger in the air. A line (initially set to red, but can be changed later on (see color selection)) will be drawn, following the path of their finger.
- Color Selection: The user can touch their index finger to their thumb to select the color of the pixel their index fingertip is currently pointing at. This will change the color of the “pen”.
- Turning pen on and off: By touching the tip of their middle finger to their thumb, the user can turn their pen on or off. The user will only be able to draw on the frame when their pen is on.
- Taking a screenshot: By touching the tip of their ring finger to their thumb, the user can take a screenshot of the current frame (including the contents drawn on the frame), which is saved in the Screenshot folder.
- Clearing the canvas: By touching the tip of their pinky finger to their thumb, all drawn lines are cleared from the frame.

In order to save a frame, the user can use the screenshot functionality (by touching the tip of their ring finger to their thumb). This saves a screenshot of the current frame to the Screenshots folder, which the user can access by clicking on the folder icon labeled as such on the upper right corner. The folder contains the collection of all screenshots the user has taken.
When the user has finished using the program, they can press the “Esc” key to exit the window and close the webcam.

