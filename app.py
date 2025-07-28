from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

app = Flask(__name__)

# Define static and model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "krones_palletizer_models.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Load models safely
try:
    models = joblib.load(MODEL_PATH)
    print("✅ Available keys in model file:", models.keys())
    model_status = models['model_status']
    model_error = models['model_code']
except Exception as e:
    print("❌ Failed to load models:", e)
    model_status = None
    model_error = None

# Error code to description mapping
error_map = {
    'E101': 'Motor overload',
    'E102': 'High temperature',
    'E103': 'Sensor failure',
    'E104': 'Emergency stop triggered',
    'E105': 'Air pressure low',
    'E106': 'Power supply failure',
    'E107': 'Communication timeout',
    'E108': 'Mechanical jam',
    'E109': 'Cooling system failure',
    'E110': 'Unknown system error'
    # Add more if needed
}

# Expected features
features = ['Temp (°C)', 'Load (%)', 'Duration (min)']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}
    if request.method == 'POST':
        try:
            if model_status is None or model_error is None:
                raise Exception("Model not loaded properly. Check model keys or file.")

            # Get input values
            temp = float(request.form['temp'])
            load = float(request.form['load'])
            duration = float(request.form['duration'])

            input_data = pd.DataFrame([[temp, load, duration]], columns=features)

            # Make predictions
            status = model_status.predict(input_data)[0]

            if status == 1:
                code = model_error.predict(input_data)[0]
                code = str(code).strip()
                description = error_map.get(code, f"⚠ Error code {code} not defined.")
            else:
                code = "None"
                description = "✅ Machine is operating normally"

            # Create bar chart
            chart_path = os.path.join(STATIC_DIR, "chart.png")
            plt.figure(figsize=(4, 2))
            plt.bar(['Temp', 'Load', 'Duration'], [temp, load, duration], color=['orange', 'green', 'blue'])
            plt.ylim(0, 100)
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()

            result = {
                'status': 'FAULT' if status == 1 else 'OK',
                'code': code,
                'desc': description,
                'chart': 'chart.png'
            }

        except Exception as e:
            print("❌ Error:", e)
            result = {
                'status': 'Error',
                'desc': f'Prediction failed: {str(e)}',
                'code': 'N/A',
                'chart': None
            }

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)