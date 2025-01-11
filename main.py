import tkinter as tk
from tkinter import ttk, messagebox
import pickle  
import numpy as np

with open('ML_Models/BaggingClassifier().pkl', 'rb') as f:
    BaggingClassifier = pickle.load(f)
with open('ML_Models/DecisionTreeClassifier().pkl', 'rb') as f:
    DecisionTreeClassifier = pickle.load(f)
with open('ML_Models/GradientBoostingClassifier().pkl', 'rb') as f:
    GradientBoostingClassifier = pickle.load(f)
with open('ML_Models/LogisticRegression().pkl', 'rb') as f:
    LogisticRegression = pickle.load(f)
with open('ML_Models/RandomForestClassifier().pkl', 'rb') as f:
    RandomForestClassifier = pickle.load(f)
with open('ML_Preprocessors/minmaxscaler.pkl', 'rb') as f:
    ms = pickle.load(f)
with open('ML_Preprocessors/standardscaler.pkl', 'rb') as f:
    sc = pickle.load(f)

models = {
    "BaggingClassifier": BaggingClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier
}

class ModelPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CropWise - Crop Recommendation System")
        self.root.geometry("800x700")

        self.bg_image = tk.PhotoImage(file="bg2.png")
        
        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        self.main_frame = tk.Frame(root, bg='#ffffff', bd=5)
        self.main_frame.place(relx=0.5, rely=0.5, relwidth=0.8, relheight=0.8, anchor='center')

        self.title_label = tk.Label(self.main_frame, text="CropWise - Crop Recommendation System", font=("LeagueSpartan Bold", 16, "bold"), bg='#ffffff')
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky='e')

        self.model_label = tk.Label(self.main_frame, text="Select Model:", bg='#ffffff', font=("LeagueSpartan", 14))
        self.model_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        
        self.model_combobox = ttk.Combobox(self.main_frame, values=list(models.keys()), font=("LeagueSpartan", 12))
        self.model_combobox.grid(row=1, column=1, padx=10, pady=10)
        self.model_combobox.current(0)

        self.entries = []
        self.labels = []
        ip_values = ['N', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        for i in range(len(ip_values)):  
            label = tk.Label(self.main_frame, text=f"{ip_values[i]}:", bg='#ffffff', font=("LeagueSpartan", 12))
            label.grid(row=i+2, column=0, padx=10, pady=10, sticky="e")
            self.labels.append(label)
            
            entry = tk.Entry(self.main_frame, font=("LeagueSpartan Bold", 12))
            entry.grid(row=i+2, column=1, padx=10, pady=10)
            self.entries.append(entry)
        
        self.predict_button = tk.Button(self.main_frame, text="Predict", command=self.predict, bg="#ff5733", fg="white", font=("LeagueSpartan", 12, "bold"))
        self.predict_button.grid(row=10, column=0, columnspan=2, padx=10, pady=20)

        self.result_label = tk.Label(self.main_frame, text="Prediction Result:", bg='#ffffff', font=("LeagueSpartan", 12))
        self.result_label.grid(row=11, column=0, columnspan=2, padx=10, pady=10)


    def predict(self):
        selected_model = self.model_combobox.get()
        model = models[selected_model]

        try:
            input_values = [float(entry.get()) for entry in self.entries]
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numbers for all input fields.")
            return

        constraints = [
            (10, 240),  # N
            (0, 750),   # K
            (5.0, 45.0), # Temperature
            (10.0, 100.0), # Humidity
            (5.0, 8.5), # pH
            (10.0, 300.0) # Rainfall
        ]

        for val, (min_val, max_val) in zip(input_values, constraints):
            if not (min_val <= val <= max_val):
                messagebox.showerror("Invalid input", f"Please enter values within the specified ranges:\n"
                                                     f"N: 10-240\nK: 0-750\n"
                                                     f"Temperature: 5-45\nHumidity: 10-100\npH: 5-8.5\nRainfall: 10-300")
                return

        input_array = np.array(input_values).reshape(1, -1)
        transformed_features = ms.transform(input_array)
        transformed_features = sc.transform(transformed_features)

        predicted_crop = model.predict(transformed_features)
        crop_dict = {1: "Rice",
             2: "Maize",
             3: "Jute",
             4: "Cotton",
             5: "Coconut",
             6: "Papaya",
             7: "Orange",
             8: "Apple",
             9: "Muskmelon",
             10: "Watermelon",
             11: "Grapes",
             12: "Mango",
             13: "Banana",
             14: "Pomegranate",
             15: "Lentil",
             16: "Blackgram",
             17: "Mungbean",
             18: "Mothbeans",
             19: "Pigeonpeas",
             20: "Kidneybeans",
             21: "Chickpea",
             22: "Coffee"
             }
        recommended_crop = crop_dict[predicted_crop[0]]
        self.result_label.config(text=f"Recommended Crop: {recommended_crop.capitalize()}")


root = tk.Tk()
app = ModelPredictorApp(root)
root.mainloop()
