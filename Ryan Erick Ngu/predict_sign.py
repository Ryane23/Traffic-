import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

IMG_WIDTH = 30
IMG_HEIGHT = 30

class TrafficSignPredictor:
    def __init__(self, model_path):
        self.load_model(model_path)
        self.categories = self.get_categories()  # Embed categories directly
        self.initialize_gui()

    def load_model(self, model_path):
        """Load the model from the specified path."""
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def get_categories(self):
        """Return a dictionary of categories."""
        return {
            "0": "Speed limit (20km/h)",
            "1": "Speed limit (30km/h)",
            "2": "Speed limit (50km/h)",
            "3": "Speed limit (60km/h)",
            "4": "Speed limit (70km/h)",
            "5": "Speed limit (80km/h)",
            "6": "End of speed limit (80km/h)",
            "7": "Speed limit (100km/h)",
            "8": "Speed limit (120km/h)",
            "9": "No passing",
            "10": "No passing for vehicles over 3.5 metric tons",
            "11": "Right-of-way at the next intersection",
            "12": "Priority road",
            "13": "Yield",
            "14": "Stop",
            "15": "No vehicles",
            "16": "Vehicles over 3.5 metric tons prohibited",
            "17": "No entry",
            "18": "General caution",
            "19": "Dangerous curve to the left",
            "20": "Dangerous curve to the right",
            "21": "Double curve",
            "22": "Bumpy road",
            "23": "Slippery road",
            "24": "Road narrows on the right",
            "25": "Road work",
            "26": "Traffic signals",
            "27": "Pedestrians",
            "28": "Children crossing",
            "29": "Bicycles crossing",
            "30": "Beware of ice/snow",
            "31": "Wild animals crossing",
            "32": "End of all speed and passing limits",
            "33": "Turn right ahead",
            "34": "Turn left ahead",
            "35": "Ahead only",
            "36": "Go straight or right",
            "37": "Go straight or left",
            "38": "Keep right",
            "39": "Keep left",
            "40": "Roundabout mandatory",
            "41": "End of no passing",
            "42": "End of no passing by vehicles over 3.5 metric tons"
        }

    def initialize_gui(self):
        """Initialize the GUI components."""
        self.window = tk.Tk()
        self.window.title("Pops Traffic Predictor")
        self.window.geometry("900x600")
        self.window.resizable(False, False)
        self.window.configure(bg="#E6F3FF")  # Light blue background

        # Create style for custom appearance
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#E6F3FF")
        style.configure("Custom.TLabel", background="#E6F3FF", font=('Segoe UI', 12))
        style.configure("Header.TLabel", background="#E6F3FF", font=('Segoe UI', 24, 'bold'))
        style.configure("Title.TLabel", background="#E6F3FF", font=('Segoe UI', 16, 'bold'))

        # Header with more padding
        header_frame = ttk.Frame(self.window, padding="20", style="Custom.TFrame")
        header_frame.pack(side=tk.TOP, fill=tk.X)
        header_label = ttk.Label(header_frame, text="Pops Traffic Predictor", style="Header.TLabel")
        header_label.pack(pady=10)

        # Main content frames with increased padding
        content_frame = ttk.Frame(self.window, padding="20", style="Custom.TFrame")
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(content_frame, padding="15", relief=tk.GROOVE)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.right_frame = ttk.Frame(content_frame, padding="15", relief=tk.GROOVE)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.setup_left_frame()
        self.setup_right_frame()

    def setup_left_frame(self):
        """Setup the left frame of the GUI."""
        self.title_label = ttk.Label(self.left_frame, text="Upload and Preview Image", style="Title.TLabel")
        self.title_label.pack(pady=15)

        # Custom style for upload button
        style = ttk.Style()
        style.configure("Upload.TButton", font=('Segoe UI', 12), padding=10)
        
        self.upload_btn = ttk.Button(self.left_frame, text="Upload Image", 
                                   command=self.upload_image, style="Upload.TButton")
        self.upload_btn.pack(pady=15)

        self.image_frame = ttk.Frame(self.left_frame, relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.pack(pady=15, fill=tk.BOTH, expand=True, padx=10)

        self.image_label = ttk.Label(self.image_frame, text="No image uploaded", 
                                   anchor=tk.CENTER, style="Custom.TLabel")
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=10)

    def setup_right_frame(self):
        """Setup the right frame of the GUI."""
        style = ttk.Style()
        style.configure("Results.TLabelframe", font=('Segoe UI', 12, 'bold'))
        style.configure("Results.TLabel", font=('Segoe UI', 12), padding=5)

        self.result_frame = ttk.LabelFrame(self.right_frame, text="Prediction Results", 
                                         padding="15", style="Results.TLabelframe")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        self.category_label = ttk.Label(self.result_frame, text="Category: ", style="Results.TLabel")
        self.category_label.pack(pady=10, anchor=tk.W)

        self.sign_label = ttk.Label(self.result_frame, text="Sign: ", style="Results.TLabel")
        self.sign_label.pack(pady=10, anchor=tk.W)

        self.confidence_label = ttk.Label(self.result_frame, text="Confidence: ", style="Results.TLabel")
        self.confidence_label.pack(pady=10, anchor=tk.W)

    def preprocess_image(self, image_path):
        """Preprocess the image for prediction."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Unable to read image from {image_path}")
        
        h, w = image.shape[:2]
        ratio = min(IMG_WIDTH / w, IMG_HEIGHT / h)
        new_size = (int(w * ratio), int(h * ratio))
        resized = cv2.resize(image, new_size)
        
        processed = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        x_offset = (IMG_WIDTH - new_size[0]) // 2
        y_offset = (IMG_HEIGHT - new_size[1]) // 2
        processed[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized  # Fixed syntax
        
        return np.expand_dims(processed, axis=0)

    def predict_sign(self, image_path):
        """Predict traffic sign from the image path."""
        try:
            processed_image = self.preprocess_image(image_path)
        except ValueError as e:
            print(e)
            return None
        
        prediction = self.model.predict(processed_image, verbose=0)
        class_id = np.argmax(prediction[0])
        probability = float(prediction[0][class_id])
        sign_name = self.categories.get(str(class_id), "Unknown")
        
        return (class_id, sign_name, probability)

    def upload_image(self):
        """Handle image upload and display prediction results."""
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                image = Image.open(file_path)
                image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo
            except Exception as e:
                print(f"Error displaying image: {e}")
                self.image_label.config(image=None)
                self.image_label.image = None
            
            result = self.predict_sign(file_path)
            if result:
                class_id, sign_name, prob = result
                self.category_label.config(text=f"Category: {class_id}")
                self.sign_label.config(text=f"Sign: {sign_name}")
                self.confidence_label.config(text=f"Confidence: {prob:.3%}")
            else:
                self.category_label.config(text="Category: N/A")
                self.sign_label.config(text="Sign: N/A")
                self.confidence_label.config(text="Confidence: N/A")

    def run(self):
        """Run the GUI application."""
        self.window.mainloop()

if __name__ == "__main__":
    MODEL_PATH = "best_model.h5"  # Update this path
    app = TrafficSignPredictor(MODEL_PATH)
    app.run()
