import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import math
from PIL import Image

# --- Configuration ---
MODEL_PATH = "model.keras" # NOTE: Using the hybrid model filename
INPUT_SIZE = (224, 224)

# WHO Haemoglobin Cut-offs (converted from g/L to g/dL)
WHO_CUTOFFS = {
    "Men (15+ years)": {"Normal": 13.0, "Mild": 11.0, "Moderate": 8.0, "Severe": 0.0},
    "Non-pregnant Women (15+ years)": {"Normal": 12.0, "Mild": 11.0, "Moderate": 8.0, "Severe": 0.0},
    "Pregnant Women": {"Normal": 11.0, "Mild": 10.0, "Moderate": 7.0, "Severe": 0.0},
    "Children 12-14 years": {"Normal": 12.0, "Mild": 11.0, "Moderate": 8.0, "Severe": 0.0},
    "Children 5-11 years": {"Normal": 11.5, "Mild": 11.0, "Moderate": 8.0, "Severe": 0.0},
    "Children 6-59 months": {"Normal": 11.0, "Mild": 10.0, "Moderate": 7.0, "Severe": 0.0},
}


# Set page configuration
st.set_page_config(
    page_title="Anemia Predictor (WHO Standards)",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Formulaic Calculation Function ---
def predict_hb_formulaic(mean_r, mean_g, mean_b):
    """
    Calculates the Hb value using the paper's formula (1) and scales it (2).
    Inputs are assumed to be mean RGB values on a 0-255 scale.
    """
    r, g, b = mean_r, mean_g, mean_b
    
    W = -1.922 + (0.206 * r) - (0.241 * g) + (0.012 * b)
    
    try:
        hb_normalized = math.exp(W) / (1 + math.exp(W))
    except OverflowError:
        hb_normalized = 1.0 
        
    min_range, max_range = 7.0, 15.0
    hb_final = min_range + (max_range - min_range) * hb_normalized
    
    return np.clip(hb_final, 7.0, 15.0)

# --- Model Loading (Cached for Efficiency) ---
@st.cache_resource
def load_hybrid_model():
    """Loads the hybrid Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model at {MODEL_PATH}. Make sure 'best_hybrid_model.keras' is in the current directory.")
        return None

# --- Image Preprocessing Function ---
def preprocess_image(uploaded_file):
    """
    Handles decoding and processing. Returns:
    1. input_tensor_cnn (Image data for CNN input)
    2. input_tensor_formula (Scalar Hb value for formula input)
    3. formula_hb_value_scalar (Scalar value for external comparison)
    """
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_raw is None:
            return None, None, None, "Error: Could not decode image data."

        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_cnn_resized = cv2.resize(img_rgb, INPUT_SIZE)
        
        # 1. CNN Input Preparation
        img_cnn_normalized = img_cnn_resized / 255.0
        input_tensor_cnn = np.expand_dims(img_cnn_normalized, axis=0)
        
        # 2. Formula Feature Calculation
        mean_r = np.mean(img_cnn_resized[:, :, 0])
        mean_g = np.mean(img_cnn_resized[:, :, 1])
        mean_b = np.mean(img_cnn_resized[:, :, 2])
        
        formula_hb_value_scalar = predict_hb_formulaic(mean_r, mean_g, mean_b)
        
        # 3. Formula Input Tensor for Hybrid Model
        input_tensor_formula = np.array([[formula_hb_value_scalar]]) 
        
        return input_tensor_cnn, input_tensor_formula, formula_hb_value_scalar, None
        
    except Exception as e:
        return None, None, None, f"An error occurred during preprocessing: {e}"

# --- Utility Functions ---

def classify_anemia(hb_value, group_key):
    """Classifies the Hb value based on WHO standards for the selected group."""
    cutoffs = WHO_CUTOFFS.get(group_key, WHO_CUTOFFS["Non-pregnant Women (15+ years)"])
    
    if hb_value < cutoffs["Severe"]:
        return "Severe Anemia", "red", "Highest Risk detected. Immediate laboratory blood test is REQUIRED.", 0.0
    elif hb_value < cutoffs["Moderate"]:
        return "Moderate Anemia", "red", "High Risk detected. Immediate consultation with a healthcare professional is strongly recommended.", cutoffs["Moderate"]
    elif hb_value < cutoffs["Mild"]:
        return "Mild Anemia", "orange", "Elevated Risk detected. Consult a doctor for confirmation and possible dietary changes.", cutoffs["Mild"]
    elif hb_value < cutoffs["Normal"]:
        return "Low-end Normal", "yellow", "Hb is slightly below the group's normal threshold. Follow-up is advised if symptoms persist.", cutoffs["Normal"]
    else:
        return "Normal/Healthy", "green", "Hb level appears within the normal range for your group.", cutoffs["Normal"]

def display_final_result(final_hb, cnn_hb, formula_hb, demographic_group):
    """Displays the final, chosen minimum Hb value and classification."""
    
    status, color, interpretation, threshold = classify_anemia(final_hb, demographic_group)

    st.markdown("---")
    st.subheader("Final Reported Hb Level")
    
    st.metric(
        label=f"Final Reported Hb (g/dL) for {demographic_group}", 
        value=f"{final_hb:.2f}"
    )
    
    st.markdown(f"**Classification (WHO Standard):** <span style='color:{color}; font-size:1.2em;'>**{status.upper()}**</span>", unsafe_allow_html=True)
    st.warning(f"**Interpretation:** {interpretation}")
    st.markdown(f"***Safety Logic Applied:*** Minimum value ($\mathbf{{{final_hb:.2f}}}$) chosen from the CNN Model ($\mathbf{{{cnn_hb:.2f}}}$) and Formula Model ($\mathbf{{{formula_hb:.2f}}}$) to maximize sensitivity.")
    st.markdown("---")


# --- Main Streamlit Function ---

def main():
    """Main Streamlit application logic."""
    st.title("🛡️ Anemia (Hb) Predictor based on WHO Standards")
    st.markdown("""
    This app uses a CNN model and a Formulaic Model, reports the **lowest** $\text{Hb}$ value for safety, and classifies the result based on your demographic using official WHO cut-offs.
    """)

    model = load_hybrid_model()
    if model is None:
        return # Stop execution if model loading failed

    # User Input Sidebar
    st.sidebar.header("Demographic Information")
    demographic_group = st.sidebar.selectbox(
        "Select your Population Group (for WHO Standards)",
        list(WHO_CUTOFFS.keys()),
        index=1 # Default to Non-pregnant Women
    )
    
    st.sidebar.info("The threshold for your selected group to be considered non-anemic is: "
                    f"**$\ge {WHO_CUTOFFS[demographic_group]['Normal']:.1f}\\,\\text{{g}}/\\text{{dL}}$**")

    # Main Content Area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a conjunctiva image (.jpg or .png)", 
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
             st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if uploaded_file is not None:
        
        if st.button("Predict & Classify Hb Level", type="primary"):
            with st.spinner('Analyzing image and calculating predictions...'):
                
                # Preprocess image and calculate both inputs/features
                input_tensor_cnn, input_tensor_formula, formula_hb_value_scalar, error = preprocess_image(uploaded_file)
                
                if error:
                    st.error(error)
                    return

                # --- 1. Hybrid Model Prediction (CNN) ---
                prediction_cnn = model.predict({
                    'image_input': input_tensor_cnn, 
                    'formula_input': input_tensor_formula
                })
                cnn_hb_value = prediction_cnn[0][0]
                
                # --- 2. Formulaic Prediction (Standalone Value) ---
                formula_hb_value = formula_hb_value_scalar

                # --- 3. MAX SAFETY LOGIC: Take the Minimum ---
                final_reported_hb = min(cnn_hb_value, formula_hb_value)

                # --- Display Results ---
                with col2:
                    st.markdown("#### Model Outputs")
                    st.info(f"CNN Model Output: **{cnn_hb_value:.2f} g/dL**")
                    st.info(f"Formulaic Model Output: **{formula_hb_value:.2f} g/dL**")

                    display_final_result(final_reported_hb, cnn_hb_value, formula_hb_value, demographic_group)

            st.markdown("""
            ***Disclaimer: This tool is for screening and educational purposes only. Always consult a licensed medical professional for accurate diagnosis and treatment.***
            """)

if __name__ == "__main__":
    main()
