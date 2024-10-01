import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import pandas as pd

# Load the best model and scaler
try:
    best_model = joblib.load(r'C:\Users\DELL PC\Desktop\AsthmaDetectionProject\best_asthma_detection_model (1).pkl')
    scaler = joblib.load(r'C:\Users\DELL PC\Desktop\AsthmaDetectionProject\scaler (1).pkl')
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    raise

def predict_asthma():
    try:
        # Get input values from the entries
        input_values = [
            float(age_entry.get()),
            float(bmi_entry.get()),
            float(lungfunctionfvc_entry.get()),
            float(lungfunctionfev1_entry.get()),
            float(pollenexposure_entry.get()),
            float(sleepquality_entry.get()),
            float(pollutionexposure_entry.get()),
            float(dietquality_entry.get()),
            float(dustexposure_entry.get()),
            float(physicalactivity_entry.get())
        ]

        # Convert the input values to a DataFrame with appropriate column names
        input_df = pd.DataFrame([input_values], columns=[
            'Age', 'BMI', 'LungFunctionFVC', 'LungFunctionFEV1', 'PollenExposure',
            'SleepQuality', 'PollutionExposure', 'DietQuality', 'DustExposure', 'PhysicalActivity'
        ])

        # Scale the input values
        input_values_scaled = scaler.transform(input_df)

        # Predict using the best model
        prediction = best_model.predict(input_values_scaled)

        # Show the result
        result = "Asthma Detected" if prediction[0] == 1 else "No Asthma"
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def create_gui():
    root = tk.Tk()
    root.title("Asthma Detection")
    root.geometry("800x600")
    root.configure(bg="lightblue")

    main_frame = tk.Frame(root, bg="lightblue")
    main_frame.pack(fill='both', expand=True, padx=20, pady=20)

    title_label = tk.Label(main_frame, text="Asthma Detection Application", font=("Arial", 24), bg="lightblue")
    title_label.pack(pady=10)

    input_frame = tk.Frame(main_frame, bg="lightblue")
    input_frame.pack(fill='x', pady=10)

    # Create entry fields manually
    def create_entry(parent, label_text):
        frame = tk.Frame(parent, bg="lightblue")
        frame.pack(fill='x', padx=10, pady=5)
        label = tk.Label(frame, text=label_text, width=20, anchor='w', bg="lightblue", font=("Arial", 14))
        label.pack(side='left')
        entry = tk.Entry(frame, font=("Arial", 14))
        entry.pack(fill='x', expand=True)
        return entry

    global age_entry, bmi_entry, lungfunctionfvc_entry, lungfunctionfev1_entry, pollenexposure_entry
    global sleepquality_entry, pollutionexposure_entry, dietquality_entry, dustexposure_entry, physicalactivity_entry

    age_entry = create_entry(input_frame, 'Age')
    bmi_entry = create_entry(input_frame, 'BMI')
    lungfunctionfvc_entry = create_entry(input_frame, 'LungFunctionFVC')
    lungfunctionfev1_entry = create_entry(input_frame, 'LungFunctionFEV1')
    pollenexposure_entry = create_entry(input_frame, 'PollenExposure')
    sleepquality_entry = create_entry(input_frame, 'SleepQuality')
    pollutionexposure_entry = create_entry(input_frame, 'PollutionExposure')
    dietquality_entry = create_entry(input_frame, 'DietQuality')
    dustexposure_entry = create_entry(input_frame, 'DustExposure')
    physicalactivity_entry = create_entry(input_frame, 'PhysicalActivity')

    button_frame = tk.Frame(main_frame, bg="lightblue")
    button_frame.pack(fill='x', pady=20)

    predict_button = tk.Button(button_frame, text="Predict", command=predict_asthma, font=("Arial", 14), bg="lightgreen")
    predict_button.pack(side='left', padx=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
