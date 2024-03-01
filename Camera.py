import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread, Event
import os
import time
import numpy as np
import pyautogui

class ScreenRecording:
    def __init__(self, output_path, stop_event):
        self.output_path = output_path
        self.screen_size = pyautogui.size()
        self.width, self.height = self.screen_size.width, self.screen_size.height
        self.fps = 30.0
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, self.fourcc, self.fps, (self.width, self.height))
        self.stop_event = stop_event

    def start_recording(self):
        try:
            while not self.stop_event.is_set():
                img = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.out.write(image)
                time.sleep(0.05)  # Sleep for 50 milliseconds (20 frames per second)

        except KeyboardInterrupt:
            pass

        finally:
            self.stop_recording()

    def stop_recording(self):
        self.out.release()
        cv2.destroyAllWindows()

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection and Video Recording")

        self.selected_camera = tk.StringVar(value="0")  # Default camera index

        self.cap = None
        self.recording = False
        self.stop_event = Event()
        self.screen_recording = None

        self.start_time = None

        # Load the classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.create_widgets()

    def create_widgets(self):
        # Camera selection
        camera_label = ttk.Label(self.root, text="Выбери камеру:")
        camera_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        camera_combobox = ttk.Combobox(self.root, textvariable=self.selected_camera, values=[str(i) for i in range(10)])
        camera_combobox.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)

        # Start/Stop buttons
        start_button = ttk.Button(self.root, text="Начать просмотр", command=self.start_detection)
        start_button.grid(row=1, column=0, padx=10, pady=10)

        stop_button = ttk.Button(self.root, text="Остановить просмотр/запись", command=self.stop_detection)
        stop_button.grid(row=1, column=1, padx=10, pady=10)

        # Record/Stop Record buttons
        record_button = ttk.Button(self.root, text="Начать запись", command=self.start_record)
        record_button.grid(row=2, column=0, padx=10, pady=10)

        # Timer label
        self.timer_label = ttk.Label(self.root, text="")
        self.timer_label.grid(row=3, column=0, columnspan=2, pady=10)

    def start_detection(self):
        camera_index = int(self.selected_camera.get())
        self.cap = cv2.VideoCapture(camera_index)

        # Set desired frame size to 1920x1080 pixels (1080p)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.recording = True
        self.record_thread = Thread(target=self.process_frames)
        self.record_thread.start()

    def stop_detection(self):
        self.recording = False
        if self.cap.isOpened():
            self.cap.release()
        if self.screen_recording is not None:
            self.stop_record()

    def start_record(self):
        if not self.recording:
            return

        self.start_time = time.time()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        today_date = time.strftime("%Y-%m-%d")
        output_path = os.path.join(script_dir, f"output_{today_date}.mp4")

        self.stop_event.clear()  # Clear the stop event flag
        self.screen_recording = ScreenRecording(output_path, self.stop_event)
        self.screen_record_thread = Thread(target=self.screen_recording.start_recording)
        self.screen_record_thread.start()

        self.root.after(1000, self.update_timer)

    def update_timer(self):
        if self.recording:
            elapsed_time = time.time() - self.start_time
            self.timer_label["text"] = f"Время записи: {int(elapsed_time)} секунд"
            self.root.after(1000, self.update_timer)

    def stop_record(self):
        if self.screen_recording is not None:
            self.stop_event.set()  # Set the stop event flag
            self.screen_record_thread.join()  # Wait for the recording thread to finish
            self.screen_recording.stop_recording()

        if self.cap.isOpened():
            self.cap.release()

        cv2.destroyAllWindows()  # Close the camera window

        self.recording = False
        self.timer_label["text"] = "Запись остановлена"
        self.root.title("Видеокамера")

    def process_frames(self):
        while self.recording:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Your face detection and processing code here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Camera", frame)

            if self.screen_recording is not None and not self.stop_event.is_set():
                self.screen_recording.out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if self.cap.isOpened():
            self.cap.release()
        if self.screen_recording is not None and not self.stop_event.is_set():
            self.screen_recording.stop_recording()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()