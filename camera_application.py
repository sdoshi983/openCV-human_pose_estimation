# Camera Application

# imported necessary library
import cv2
import time
import threading
from cv2 import cv2
from PIL import Image, ImageTk
from tkinter import Label, Button, Tk, PhotoImage
from tkinter import filedialog
import tkinter.messagebox as mbox
import argparse

# created a class for camera application where main application is created
class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Camera Application")
        self.window.geometry("1000x700")
        # self.window.configure(bg="gray")
        self.window.resizable(1, 1)
        Label(self.window, width=1000, height=600, bg="light yellow").place(x=0, y=320)
        # self.topLabel = Label(self.window, text = "CAMERA" , font=("Arial", 40),fg="magenta", bg="light green")
        self.TakePhoto_b = Button(self.window, text="Take a Shot", font=("Arial", 20), bg="light green", fg = "blue", relief='raised',command=self.TakePhoto)
        self.see_b = Button(self.window, text="SEE THIS", font=("Arial", 20), bg="orange", fg = "blue", relief='raised',command=self.see_this)

        # self.prev_b = Button(self.window, text="PREVIEW", font=("Arial", 20), bg="orange", fg = "blue", relief='raised',command=self.prev_img)
        self.exit_b = Button(self.window, text="EXIT", font=("Arial", 20), bg="red", fg = "blue", relief='raised',command=self.exit_win)
        self.ImageLabel = Label(self.window, width=1000, height=500, bg="light yellow")
        self.ImageLabel.place(x=0, y=0)
        self.TakePhoto_b.place(x=150, y=560)
        # self.prev_b.place(x=440, y=560)
        self.see_b.place(x=440, y=560)
        self.exit_b.place(x = 750, y = 560)
        # self.topLabel.place(x = 250, y = 20)
        self.take_picture = False
        self.PictureTaken = False
        self.Main()

    # method for loading the camera
    @staticmethod
    def LoadCamera():
        camera = cv2.VideoCapture("assets/sample.mp4")
        camera.set(cv2.CAP_PROP_FPS, 1)   # setting fps

        camera.set(3, 800)
        camera.set(4, 800)

        parser = argparse.ArgumentParser()
        parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
        parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
        parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
        parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

        args = parser.parse_args()

        net = cv2.dnn.readNetFromTensorflow('assets/graph_opt.pb')

        inWidth = 368
        inHeight = 368
        thr = 0.2

        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

        POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


        #    cv2.imshow('Human Body', frame)
        if camera.isOpened():
            bool, frame = camera.read()
        while bool:
            bool, frame = camera.read()

            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            net.setInput(
                cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))

            out = net.forward()
            out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

            assert (len(BODY_PARTS) == out.shape[1])

            points = []
            for i in range(len(BODY_PARTS)):
                # Slice heatmap of corresponging body's part.
                heatMap = out[0, i, :, :]

                # Originally, we try to find all the local maximums. To simplify a sample
                # we just find a global one. However only a single pose at the same time
                # could be detected this way.
                _, conf, _, point = cv2.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                # Add a point if it's confidence is higher than threshold.
                points.append((int(x), int(y)) if conf > args.thr else None)

            for pair in POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                assert (partFrom in BODY_PARTS)
                assert (partTo in BODY_PARTS)

                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]

                if points[idFrom] and points[idTo]:
                    cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                    cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

            t, _ = net.getPerfProfile()
            freq = cv2.getTickFrequency() / 1000
            cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            if bool:
                yield frame
            else:
                yield False
            if not bool:
                cv2.waitKey()
                break

    # method for exiting the window
    def exit_win(self):
        if mbox.askokcancel("Exit", "Do you want to exit?"):
            self.window.destroy()

    # def prev_img(self):
    #     self.window1 = Tk()
    #     self.window1.title("Your Image")
    #     self.window1.geometry("1000x700")
    #
    #     # image on the main window
    #     path = "myimage.png"
    #     # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    #     img1 = ImageTk.PhotoImage(Image.open(path))
    #     # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    #     panel = Label(self.window1, image=img1)
    #     panel.place(x=260, y=250)

    # method to show how camera works
    def see_this(self):
        mbox.showinfo("Details", "When you click on Take a Shot, It will show a message of taking image.\n\nWhen we click on OK, it will ask user to save at any location in local system.\n\nAfter saving, it will show message, that File Saved Successfully.\n\nAfter that when user click on Take Again button, it willl give a message og reconfiguring camera, and clicking on the OK will start the camera and user can take the shot again.")

    # method defined to take the photo
    def TakePhoto(self):
        if not self.PictureTaken:
            # print('Taking a Picture')
            mbox.showinfo("Status", "Taking Picture")
            self.take_picture = True
        else:
            # print("Reconfiguring camera")
            mbox.showinfo("Status", "Reconfiguring camera")
            self.TakePhoto_b.configure(text="Take a Shot")
            self.take_picture = False

    # main method defined
    def Main(self):
        self.render_thread = threading.Thread(target=self.StartCamera)
        self.render_thread.daemon = True
        self.render_thread.start()

    # method to start the camera
    def StartCamera(self):
        frame = self.LoadCamera()
        CaptureFrame = None
        while True:
            if frame:
                Frame = next(frame)
                # print(self.take_picture)
                if frame and not self.take_picture:
                    picture = Image.fromarray(Frame)
                    picture = picture.resize((700, 450), resample=0)
                    CaptureFrame = picture.copy()
                    picture = ImageTk.PhotoImage(picture)
                    self.ImageLabel.configure(image=picture)
                    self.ImageLabel.photo = picture
                    self.PictureTaken = False
                    time.sleep(0.001)
                else:
                    if not self.PictureTaken:
                        # print("Your camera died")
                        # mbox.showinfo("Status","Your camera died")

                        CaptureFrame.save('myimage.png')

                        img = cv2.imread('myimage.png')
                        edge = Image.fromarray(img)
                        filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
                        if not filename:
                            mbox.showinfo("Success", "Image not saved!")
                            break
                        else:
                            edge.save(filename)
                        mbox.showinfo("Success", "Image Saved Successfully!")

                        self.TakePhoto_b.configure(text="Take Again")
                        self.PictureTaken = True

# function defined for exiting the window
def exit_win1():
    if mbox.askokcancel("Exit", "Do you want to exit?"):
        root.destroy()

root = Tk()
App = CameraApp(root)
root.protocol("WM_DELETE_WINDOW", exit_win1)
root.mainloop()
