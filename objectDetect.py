import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import math

class WebcamApp:
    def __init__(self, window, window_title="Webcam App"):
        self.window = window
        self.window.title(window_title)

        

        # Open a connection to the webcam (camera index 0 by default)
        self.cap = cv2.VideoCapture(0)
        
        # Check if the webcam is opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Initialize variables
        self.captured_image = None

        # Create Canvas to display the webcam feed
        self.canvas = tk.Canvas(window, width=self.cap.get(3), height=self.cap.get(4))
        self.canvas.pack()

        # Create Capture button
        self.capture_button = tk.Button(window, text="Capture", command=self.capture_image)
        self.capture_button.pack(pady=10)

        # Create labels to display image metrics
        self.mean_label = tk.Label(window, text="Mean Intensity:")
        self.mean_label.pack()

        self.contrast_label = tk.Label(window, text="Contrast:")
        self.contrast_label.pack()

        self.saturation_label = tk.Label(window, text="Saturation:")
        self.saturation_label.pack()

        self.brightness_label = tk.Label(window, text="Brightness:")
        self.brightness_label.pack()

        self.sharpness_label = tk.Label(window, text="Sharpness:")
        self.sharpness_label.pack()

        self.color_balance_label = tk.Label(window, text="Color Balance:")
        self.color_balance_label.pack()

        # Create Exit button
        self.exit_button = tk.Button(window, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=10)

        # Load YOLO model
        self.model = YOLO("yolo-Weights/yolov8n.pt")

        # Object classes
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                      "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                      "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"
                      ]

        # Update GUI
        self.update()

    def update(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()

        # Display the frame on the Canvas
        if ret:
            # Convert frame to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Calculate and display image metrics
            if self.captured_image is not None:
                metrics = self.calculate_image_metrics(frame)
                self.update_metric_labels(metrics)

            # Detect objects and draw bounding boxes
            self.detect_and_draw_objects(frame)

        # Repeat the update after 10 milliseconds
        self.window.after(10, self.update)

    def calculate_image_metrics(self, frame):
        # Convert frame to grayscale for some metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate mean intensity
        mean_intensity = np.mean(frame)

        # Calculate contrast
        contrast = np.std(gray)

        # Calculate saturation (not highly accurate)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:,:,1])

        # Calculate brightness
        brightness = np.mean(frame)

        # Calculate sharpness (edge detection)
        edges = cv2.Canny(gray, 50, 150)
        sharpness = np.mean(edges)

        # Calculate color balance
        b, g, r = cv2.split(frame)
        total_pixels = frame.shape[0] * frame.shape[1]
        color_balance = (np.sum(b) / total_pixels, np.sum(g) / total_pixels, np.sum(r) / total_pixels)

        return {
            "mean_intensity": mean_intensity,
            "contrast": contrast,
            "saturation": saturation,
            "brightness": brightness,
            "sharpness": sharpness,
            "color_balance": color_balance
        }

    def update_metric_labels(self, metrics):
        self.mean_label.config(text=f"Mean Intensity: {metrics['mean_intensity']:.2f}")
        self.contrast_label.config(text=f"Contrast: {metrics['contrast']:.2f}")
        self.saturation_label.config(text=f"Saturation: {metrics['saturation']:.2f}")
        self.brightness_label.config(text=f"Brightness: {metrics['brightness']:.2f}")
        self.sharpness_label.config(text=f"Sharpness: {metrics['sharpness']:.2f}")
        self.color_balance_label.config(text=f"Color Balance: R={metrics['color_balance'][2]:.2f}, G={metrics['color_balance'][1]:.2f}, B={metrics['color_balance'][0]:.2f}")

    def detect_and_draw_objects(self, frame):
        results = self.model(frame, stream=True)

        # Coordinates and information about detected objects
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # Class name
                cls = int(box.cls[0])
                class_name = self.classNames[cls]

                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, f"{class_name} {confidence}", org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) == ord('q'):
            self.exit_app()

    def capture_image(self):
        # Capture an image
        ret, frame = self.cap.read()

        # Save the captured image
        if ret:
            self.captured_image = frame.copy()

    def exit_app(self):
        # Release the webcam and close the window
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
