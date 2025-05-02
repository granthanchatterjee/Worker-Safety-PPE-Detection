import os
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"
TEST_IMAGE_FOLDER = "datasets/datasets/images"

CLASS_NAMES = [
    "person",
    "hard-hat",
    "gloves",
    "mask",
    "glasses",
    "boots",
    "vest",
    "ppe-suit",
    "ear-protector",
    "safety-harness"
]

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    exit(1)

test_images = [f for f in os.listdir(TEST_IMAGE_FOLDER)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not test_images:
    print("No test images found in", TEST_IMAGE_FOLDER)
    exit(1)

class DetectionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Detection Viewer")

        self.current_index = 0

        self.image_panel = tk.Label(master)
        self.image_panel.pack()

        self.filter_frame = tk.Frame(master)
        self.filter_frame.pack(pady=5)

        self.filter_vars = {}
        for cls in CLASS_NAMES:
            var = tk.IntVar(value=1)
            self.filter_vars[cls] = var
            cb = tk.Checkbutton(self.filter_frame, text=cls,
                                variable=var, command=self.update_image)
            cb.pack(side=tk.LEFT, padx=2)

        self.select_frame = tk.Frame(master)
        self.select_frame.pack(pady=5)
        self.select_all_button = ttk.Button(
            self.select_frame, text="Select All", command=self.select_all)
        self.select_all_button.pack(side=tk.LEFT, padx=5)
        self.deselect_all_button = ttk.Button(
            self.select_frame, text="Deselect All", command=self.deselect_all)
        self.deselect_all_button.pack(side=tk.LEFT, padx=5)

        nav_frame = tk.Frame(master)
        nav_frame.pack(pady=5)
        self.prev_button = ttk.Button(nav_frame, text="<< Prev",
                                      command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=10)
        self.next_button = ttk.Button(nav_frame, text="Next >>",
                                      command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=10)

        self.update_image()

    def select_all(self):
        """Enable all class filters."""
        for cls in CLASS_NAMES:
            self.filter_vars[cls].set(1)
        self.update_image()

    def deselect_all(self):
        """Disable all class filters."""
        for cls in CLASS_NAMES:
            self.filter_vars[cls].set(0)
        self.update_image()

    def update_image(self):

        img_path = os.path.join(TEST_IMAGE_FOLDER, test_images[self.current_index])
        orig_image = cv2.imread(img_path)
        if orig_image is None:
            print("Error reading image:", img_path)
            return

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        results = model(image)
        boxes = results[0].boxes

        disp_image = image.copy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"

            if self.filter_vars[label].get() == 1:
                cv2.rectangle(disp_image, (x1, y1), (x2, y2),
                              (255, 0, 0), 2)
                cv2.putText(disp_image, f"{label} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)

        from PIL import Image
        pil_image = Image.fromarray(disp_image)
        pil_image = pil_image.resize((800, 600))
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_panel.configure(image=tk_image)
        self.image_panel.image = tk_image

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(test_images)
        self.update_image()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(test_images)
        self.update_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop()