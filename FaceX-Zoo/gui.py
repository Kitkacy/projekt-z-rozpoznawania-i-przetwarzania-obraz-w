from tkinter import *
from tkinter import ttk
from tkinter import filedialog, messagebox
import cv2
import threading
from live_face import LiveFace

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facio Recognitio")
        self.root.grid()

        

        # grid 3x1
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.columnconfigure(3, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)


        self.source_label = ttk.Label(root, text="Wybierz źródło obrazu:")
        self.source_label.grid(row=1, column=0, sticky="sw", padx=20, pady=10)

        self.option_label = ttk.Label(root, text="Dodatkowe opcje:")
        self.option_label.grid(row=2, column=0, sticky="sw", padx=20, pady=10)


        self.choose_btn = ttk.Button(root, text="Z pliku wideo", command=self.choose_file)
        self.choose_btn.grid(row=2, column=0, sticky="n")

        self.live_btn = ttk.Button(root, text="Na żywo", command=self.live_video)
        self.live_btn.grid(row=2, column=1, sticky="nw")

        self.config_btn = ttk.Button(root, text="Konfigurator", command=self.open_configurator)
        self.config_btn.grid(row=3, column=0, sticky="n")

        self.record_btn = ttk.Button(root, text="Rozpocznij nagrywanie", command=self.start_recording)
        self.record_btn.grid(row=3, column=1, sticky="nw")

        self.video_source = 0
        self.from_file = 0
        self.cap = None
        self.recording = False
        self.output_file = "output.mp4"

    def choose_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pliki wideo", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_source = file_path
            # uruchomienie rozpoznawania twarzy
            live_face = LiveFace(self.video_source)
            live_face.run()


    def open_configurator(self):
        messagebox.showinfo("Konfigurator", "Tutaj pojawi się konfigurator (do zaimplementowania).")

    def start_recording(self):
        self.recording = True
        threading.Thread(target=self.record_video, daemon=True).start()

    def record_video(self):
        cam = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_file, fourcc, 20.0, (frame_width, frame_height))

        while cam.isOpened() and self.recording:
            ret, frame = cam.read()
            if not ret:
                break
            out.write(frame)
            cv2.imshow('Nagrywanie', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        out.release()
        cv2.destroyAllWindows()
        self.recording = False
    
    def live_video(self):
        # if self.video_source == 0:
        #     return
        # live_face = LiveFace(self.video_source)
        # live_face.run()
        live_face = LiveFace(self.video_source)
        live_face.run_live()


if __name__ == "__main__":
    root = Tk()
    app = VideoApp(root)
    root.geometry("640x480")
    root.mainloop()
