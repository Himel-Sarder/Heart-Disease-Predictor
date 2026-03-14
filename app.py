import gradio as gr
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Load the trained model
model = CatBoostClassifier()
model.load_model("catboost_heart_model.cbm")

# Define feature names in the correct order
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Create description for each feature
feature_descriptions = {
    'age': 'Age in years',
    'sex': 'Biological sex',
    'cp': 'Type of chest pain experienced',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
    'restecg': 'Resting electrocardiographic results',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of the peak exercise ST segment',
    'ca': 'Number of major vessels colored by fluoroscopy',
    'thal': 'Thalassemia type'
}

# Mapping dictionaries for categorical variables
sex_mapping = {"Female": 0, "Male": 1}

cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1, 
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

restecg_mapping = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

slope_mapping = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_mapping = {
    "Normal": 0,
    "Fixed Defect": 1, 
    "Reversible Defect": 2
}

# Advice based on prediction
def get_advice(prediction, probability):
    if prediction == 1:
        return """
        **IMPORTANT MEDICAL ADVICE:**
        • Please consult a cardiologist immediately
        • Schedule a complete cardiac evaluation
        • Consider lifestyle modifications: diet, exercise, stress management
        • Monitor blood pressure and cholesterol regularly
        • Take prescribed medications as directed by your doctor
        
        *This is an AI prediction tool. Always seek professional medical advice.*
        """
    else:
        return """
        **PREVENTIVE HEALTH ADVICE:**
        • Maintain a healthy diet low in saturated fats
        • Exercise regularly (150 minutes moderate activity per week)
        • Avoid smoking and limit alcohol consumption
        • Regular health check-ups annually
        • Manage stress through meditation or yoga
        
        *Continue healthy habits to maintain good heart health.*
        """

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, 
                          thalach, exang, oldpeak, slope, ca, thal):
    """
    Predict heart disease based on input features
    """
    # Convert all categorical inputs to numerical values
    sex_numeric = sex_mapping[sex]
    cp_numeric = cp_mapping[cp]
    restecg_numeric = restecg_mapping[restecg]
    slope_numeric = slope_mapping[slope]
    thal_numeric = thal_mapping[thal]
    fbs_numeric = 1 if fbs else 0
    exang_numeric = 1 if exang else 0
    
    # Create dataframe with input values
    input_data = pd.DataFrame([[age, sex_numeric, cp_numeric, trestbps, chol, fbs_numeric, 
                                 restecg_numeric, thalach, exang_numeric, oldpeak, 
                                 slope_numeric, ca, thal_numeric]], 
                              columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Create result message
    if prediction == 1:
        result = "HIGH RISK of Heart Disease"
        confidence = probability[1] * 100
    else:
        result = "LOW RISK of Heart Disease"
        confidence = probability[0] * 100
    
    # Get advice
    advice = get_advice(prediction, probability)
    
    return result, f"Confidence: {confidence:.2f}%", advice

# Create the Gradio interface with centered layout
with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        <div style='text-align: center;'>
            <h1>Heart Disease Prediction App</h1>
            <h3>Enter patient information to predict the risk of heart disease</h3>
            <p style='color: #666;'>Model Accuracy: 98.5% | Precision: 0.99 | Recall: 0.99</p>
        </div>
        """
    )
    
    with gr.Column(variant="panel"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Patient Information")
                
                age = gr.Number(
                    label="Age", 
                    value=52, 
                    minimum=20, 
                    maximum=100,
                    info=feature_descriptions['age']
                )
                
                sex = gr.Dropdown(
                    choices=["Female", "Male"], 
                    label="Sex", 
                    value="Female", 
                    info=feature_descriptions['sex']
                )
                
                cp = gr.Dropdown(
                    choices=list(cp_mapping.keys()),
                    label="Chest Pain Type", 
                    value="Typical Angina", 
                    info=feature_descriptions['cp']
                )
                
                trestbps = gr.Number(
                    label="Resting Blood Pressure", 
                    value=120, 
                    minimum=80, 
                    maximum=200,
                    info=feature_descriptions['trestbps']
                )
                
                chol = gr.Number(
                    label="Cholesterol", 
                    value=200, 
                    minimum=100, 
                    maximum=600,
                    info=feature_descriptions['chol']
                )
                
                fbs = gr.Checkbox(
                    label="Fasting Blood Sugar > 120 mg/dl", 
                    value=False, 
                    info=feature_descriptions['fbs']
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Clinical Measurements")
                
                restecg = gr.Dropdown(
                    choices=list(restecg_mapping.keys()),
                    label="Resting ECG Results", 
                    value="Normal", 
                    info=feature_descriptions['restecg']
                )
                
                thalach = gr.Number(
                    label="Max Heart Rate", 
                    value=150, 
                    minimum=60, 
                    maximum=220,
                    info=feature_descriptions['thalach']
                )
                
                exang = gr.Checkbox(
                    label="Exercise Induced Angina", 
                    value=False, 
                    info=feature_descriptions['exang']
                )
                
                oldpeak = gr.Number(
                    label="ST Depression", 
                    value=1.0, 
                    minimum=0, 
                    maximum=10,
                    step=0.1,
                    info=feature_descriptions['oldpeak']
                )
                
                slope = gr.Dropdown(
                    choices=list(slope_mapping.keys()),
                    label="ST Slope", 
                    value="Flat", 
                    info=feature_descriptions['slope']
                )
                
                ca = gr.Slider(
                    minimum=0, 
                    maximum=3, 
                    step=1, 
                    label="Major Vessels", 
                    value=0, 
                    info=feature_descriptions['ca']
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Additional Parameters")
                thal = gr.Dropdown(
                    choices=list(thal_mapping.keys()), 
                    label="Thalassemia", 
                    value="Normal", 
                    info=feature_descriptions['thal']
                )
        
        with gr.Row():
            predict_btn = gr.Button(
                "Predict Heart Disease Risk", 
                variant="primary", 
                size="lg"
            )
        
        with gr.Row():
            with gr.Column():
                result_output = gr.Textbox(
                    label="Prediction Result",
                    lines=2,
                    show_label=True,
                    elem_classes="result-text"
                )
                
                confidence_output = gr.Textbox(
                    label="Confidence Level",
                    lines=1,
                    show_label=True
                )
        
        with gr.Row():
            advice_output = gr.Markdown(
                label="Health Advice",
                value="Click 'Predict Heart Disease Risk' to see personalized health advice"
            )
        
        gr.Markdown(
            """
            ---
            <div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
                <strong>Disclaimer:</strong> This tool is for educational purposes only. 
                Always consult with a healthcare professional for medical advice.
            </div>
            """
        )
    
    # CSS for styling
    gr.HTML("""
    <style>
        .gradio-button {
            margin: 0 auto;
            display: block;
            width: 300px;
        }
        .result-text {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
        .gr-box {
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .gr-markdown h3 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
            margin-top: 10px;
        }
        button.primary {
            background-color: #3498db !important;
            color: white !important;
        }
        button.primary:hover {
            background-color: #2980b9 !important;
        }
    </style>
    """)
    
    # Set up prediction function
    predict_btn.click(
        fn=predict_heart_disease,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                exang, oldpeak, slope, ca, thal],
        outputs=[result_output, confidence_output, advice_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()