import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pickle
import time
import io
import traceback
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Dataset path - using relative path for deployment environment
DATASET_PATH = "Potato"  # Expected to be in the same directory as the app
IMG_SIZE = (224, 224)
NUM_CLASSES = None
CLASS_NAMES = []

# Load and extract features
def load_and_extract_features():
    global NUM_CLASSES, CLASS_NAMES
    
    # Initialize progress bar
    progress_text = "Loading and extracting features..."
    progress_bar = st.progress(0)
    st.text(progress_text)
    
    X, y = [], []
    CLASS_NAMES = []
    
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset path not found: {DATASET_PATH}")
        st.info("Please make sure the 'Potato' folder with disease subfolders is in the same directory as this app.")
        return None, None
    
    try:
        # Get all subdirectories (classes)
        class_dirs = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        
        for idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(DATASET_PATH, class_name)
            if os.path.isdir(class_dir):
                CLASS_NAMES.append(class_name)
                
                # Get all image files in the class directory
                img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_files = len(img_files)
                
                for i, img_file in enumerate(img_files):
                    # Update progress
                    progress = (idx * total_files + i) / (len(class_dirs) * total_files)
                    progress_bar.progress(progress)
                    
                    img_path = os.path.join(class_dir, img_file)
                    img = Image.open(img_path).convert('L').resize(IMG_SIZE)  # Convert to grayscale
                    img_array = np.array(img)
                    
                    # Extract HOG features
                    fd = hog(img_array, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=False)
                    
                    X.append(fd)
                    y.append(idx)
        
        NUM_CLASSES = len(CLASS_NAMES)
        progress_bar.progress(1.0)
        
        if NUM_CLASSES == 0:
            st.error("No classes found in the dataset.")
            return None, None
            
        st.success(f"Features extracted: {len(X)} images, {NUM_CLASSES} classes")
        st.write(f"Classes found: {', '.join(CLASS_NAMES)}")
        
        return np.array(X), np.array(y)
        
    except Exception as e:
        st.error(f"Error in feature extraction: {str(e)}")
        st.code(traceback.format_exc())
        return None, None

# Train and evaluate ML models
def train_and_evaluate(X, y):
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        st.error("Cannot train models: No valid features extracted")
        return None
    
    try:
        st.text("Training machine learning models...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Progress bar for model training
        progress_bar = st.progress(0)
        
        # SVM model
        progress_bar.progress(0.1)
        st.text("Training SVM model...")
        svm = SVC(kernel='linear', random_state=42, probability=True)
        svm.fit(X_train, y_train)
        progress_bar.progress(0.4)
        svm_pred = svm.predict(X_test)
        svm_report = classification_report(y_test, svm_pred, target_names=CLASS_NAMES, output_dict=True)
        svm_cm = confusion_matrix(y_test, svm_pred)
        
        # Random Forest model
        progress_bar.progress(0.5)
        st.text("Training Random Forest model...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        progress_bar.progress(0.8)
        rf_pred = rf.predict(X_test)
        rf_report = classification_report(y_test, rf_pred, target_names=CLASS_NAMES, output_dict=True)
        rf_cm = confusion_matrix(y_test, rf_pred)
        
        progress_bar.progress(1.0)
        st.success("Model training completed!")
        
        models = {
            'SVM': svm,
            'Random Forest': rf
        }
        
        return {
            'models': models,
            'SVM': {'model': svm, 'report': svm_report, 'cm': svm_cm},
            'Random Forest': {'model': rf, 'report': rf_report, 'cm': rf_cm},
            'test_data': (X_test, y_test)
        }
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        st.code(traceback.format_exc())
        return None

# Plot confusion matrices
def plot_confusion_matrices(cms):
    figures = {}
    for name, cm in cms.items():
        fig = go.Figure(data=go.Heatmap(
            z=cm, 
            x=CLASS_NAMES, 
            y=CLASS_NAMES, 
            colorscale='Blues', 
            text=cm, 
            texttemplate="%{text}"
        ))
        fig.update_layout(
            title=f'Confusion Matrix - {name}', 
            xaxis_title='Predicted', 
            yaxis_title='True',
            height=500
        )
        figures[name] = fig
    return figures

# Plot model performance metrics
def plot_model_performance(reports):
    metrics = ['precision', 'recall', 'f1-score']  
    models = list(reports.keys())
    data = []
    
    for metric in metrics:
        for model in models:
            data.append({
                'Model': model, 
                'Metric': metric, 
                'Value': reports[model]['weighted avg'][metric]
            })
    
    fig = px.bar(
        data, 
        x='Model', 
        y='Value', 
        color='Metric', 
        barmode='group', 
        title='Model Performance Comparison',
        height=500
    )
    fig.update_layout(yaxis_title='Score', yaxis=dict(range=[0, 1]))
    return fig

# Function to process user-uploaded image for prediction
def process_uploaded_image(image_file, models):
    if image_file is None or models is None:
        return None
    
    try:
        # Load and preprocess the image
        image = Image.open(image_file).convert('L').resize(IMG_SIZE)
        
        # Extract HOG features
        img_array = np.array(image)
        features = hog(img_array, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), visualize=False)
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        results = {}
        for name, model in models.items():
            # Get prediction
            prediction = model.predict(features)[0]
            
            # Get probability scores if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                class_probabilities = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}
            else:
                class_probabilities = {class_name: 0.0 for class_name in CLASS_NAMES}
                class_probabilities[CLASS_NAMES[prediction]] = 1.0
            
            results[name] = {
                'prediction': CLASS_NAMES[prediction],
                'probabilities': class_probabilities
            }
        
        return results
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")
        st.code(traceback.format_exc())
        return None

# Main Streamlit app
def main():
    # Declare global variables at the beginning
    global CLASS_NAMES, NUM_CLASSES
    
    try:
        st.set_page_config(
            page_title="Plant Disease Detection", 
            page_icon="🌿", 
            layout="wide"
        )
        
        st.title("🌿 Plant Disease Detection")
        st.write("""
        This application uses machine learning to detect diseases in plants from leaf images.
        Currently the models are trained to recognize potato plant diseases: Early Blight, Late Blight, and Healthy leaves.
        """)
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Project Overview", "Model Training & Evaluation", "Disease Prediction"])
        
        # Tab 1: Project Overview
        with tab1:
            st.header("Project Overview")
            
            st.subheader("Significance")
            st.write("""
            Early detection of plant diseases is critical for:
            - Effective disease management
            - Reducing crop losses
            - Ensuring food security
            - Reducing pesticide usage
            - Improving farm productivity
            """)
            
            st.subheader("Project Goal")
            st.write("""
            Develop an automated system for quick and accurate plant disease detection using 
            machine learning models. This helps farmers identify diseases early and take appropriate actions.
            """)
            
            st.subheader("Dataset Description")
            st.write("""
            The dataset consists of images of diseased and healthy potato leaves organized in three categories:
            1. **Potato___Early_blight** - Leaves affected by early blight disease
            2. **Potato___Late_blight** - Leaves affected by late blight disease
            3. **Potato___healthy** - Healthy potato leaves
            
            The images are in JPEG/PNG format, and the dataset is structured in folders, with each 
            folder representing a class (disease category).
            """)
            
            st.subheader("Machine Learning Approach")
            st.write("""
            1. **Feature Extraction**: Histogram of Oriented Gradients (HOG) to capture leaf texture and patterns
            2. **Classification Models**:
               - Support Vector Machine (SVM) with linear kernel
               - Random Forest Classifier
            3. **Evaluation**: Performance metrics including accuracy, precision, recall, and F1-score
            """)
        
        # Tab 2: Model Training & Evaluation
        with tab2:
            st.header("Model Training & Evaluation")
            
            # Check if dataset exists
            if not os.path.exists(DATASET_PATH):
                st.error(f"Dataset path not found: {DATASET_PATH}")
                st.info("""
                Please make sure the 'Potato' folder with disease subfolders is in the same directory as this app.
                
                The expected folder structure is:
                - Potato/
                  - Potato___Early_blight/
                  - Potato___healthy/
                  - Potato___Late_blight/
                """)
            else:
                st.info(f"Dataset found at: {DATASET_PATH}")
                
                results_file = "results_ml.pkl"
                train_models = st.button("Train Models", use_container_width=True)
                
                # Check if model results already exist and user hasn't requested retraining
                
                if os.path.exists(results_file) and not train_models:
                    st.info("Loading saved ML results...")
                    try:
                        with open(results_file, 'rb') as f:
                            results = pickle.load(f)
                            # Update global variables in case they weren't set
                            if 'class_names' in results:
                                CLASS_NAMES = results['class_names']
                                NUM_CLASSES = len(CLASS_NAMES)
                            st.success("Models loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading saved results: {str(e)}")
                        train_models = True  # Force retraining on error
                
                # Train models if requested or if no saved models exist
                if train_models or not os.path.exists(results_file):
                    st.write("Starting feature extraction and model training...")
                    
                    # Extract features
                    X, y = load_and_extract_features()
                    
                    if X is not None and y is not None and len(X) > 0 and len(y) > 0:
                        # Train and evaluate models
                        results = train_and_evaluate(X, y)
                        
                        if results is not None:
                            # Save results
                            try:
                                # Add class names to results
                                results['class_names'] = CLASS_NAMES
                                
                                with open(results_file, 'wb') as f:
                                    pickle.dump(results, f)
                                st.success("Results saved for future use.")
                            except Exception as e:
                                st.warning(f"Could not save model results: {str(e)}")
                    else:
                        st.error("Feature extraction failed. Cannot train models.")
                        results = None
                
                # If we have results, display model performance
                if 'results' in locals() and results is not None:
                    # Get reports and confusion matrices
                    reports = {k: v['report'] for k, v in results.items() if k in ['SVM', 'Random Forest']}
                    cms = {k: v['cm'] for k, v in results.items() if k in ['SVM', 'Random Forest']}
                    
                    # Plot model performance comparison
                    st.subheader("Model Performance Comparison")
                    perf_fig = plot_model_performance(reports)
                    st.plotly_chart(perf_fig, use_container_width=True)
                    
                    # Display confusion matrices
                    st.subheader("Confusion Matrices")
                    cm_figs = plot_confusion_matrices(cms)
                    
                    # Create two columns for the confusion matrices
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(cm_figs['SVM'], use_container_width=True)
                    with col2:
                        st.plotly_chart(cm_figs['Random Forest'], use_container_width=True)
                    
                    # Show detailed classification reports
                    st.subheader("Detailed Classification Reports")
                    
                    # Format and display reports in expanders
                    for model_name, report in reports.items():
                        with st.expander(f"{model_name} Classification Report"):
                            # Create a prettier display of the classification report
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
        
        # Tab 3: Disease Prediction
        with tab3:
            st.header("Disease Prediction")
            
            # Check if models exist
            results_file = "results_ml.pkl"
            if not os.path.exists(results_file):
                st.warning("No trained models found. Please go to the 'Model Training & Evaluation' tab and train the models first.")
            else:
                # Proceed with model loading and prediction
                
                try:
                    # Load saved models
                    with open(results_file, 'rb') as f:
                        results = pickle.load(f)
                    
                    # Update global variables
                    if 'class_names' in results:
                        CLASS_NAMES = results['class_names']
                        NUM_CLASSES = len(CLASS_NAMES)
                    
                    # Get models for prediction
                    if 'models' in results:
                        models = results['models']
                    else:
                        models = {
                            'SVM': results['SVM']['model'],
                            'Random Forest': results['Random Forest']['model']
                        }
                    
                    st.write("""
                    Upload an image of a plant leaf to detect if it has any diseases.
                    
                    **Note**: For best results, upload clear images of leaves against a plain background.
                    Currently trained on potato leaves (Early Blight, Late Blight, and Healthy).
                    """)
                    
                    # File uploader
                    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
                    
                    if uploaded_file is not None:
                        # Display the uploaded image
                        image = Image.open(uploaded_file)
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(image, caption="Uploaded Image", use_container_width=True)
                        
                        with col2:
                            with st.spinner("Processing image..."):
                                # Process the image and get predictions
                                predictions = process_uploaded_image(uploaded_file, models)
                                
                                if predictions:
                                    st.subheader("Predictions")
                                    
                                    # Get SVM results
                                    svm_pred = predictions['SVM']['prediction']
                                    svm_probas = predictions['SVM']['probabilities']
                                    
                                    # Get Random Forest results
                                    rf_pred = predictions['Random Forest']['prediction']
                                    rf_probas = predictions['Random Forest']['probabilities']
                                    
                                    # Display predictions in a table
                                    pred_df = pd.DataFrame({
                                        'Model': ['SVM', 'Random Forest'],
                                        'Prediction': [svm_pred, rf_pred]
                                    })
                                    st.dataframe(pred_df, use_container_width=True)
                                    
                                    # Create bar charts for probability distributions
                                    st.subheader("Prediction Probabilities")
                                    
                                    # Prepare data for SVM probabilities
                                    svm_proba_data = [
                                        {'Class': class_name, 'Probability': prob, 'Model': 'SVM'} 
                                        for class_name, prob in svm_probas.items()
                                    ]
                                    
                                    # Prepare data for Random Forest probabilities
                                    rf_proba_data = [
                                        {'Class': class_name, 'Probability': prob, 'Model': 'Random Forest'} 
                                        for class_name, prob in rf_probas.items()
                                    ]
                                    
                                    # Combine data
                                    all_proba_data = svm_proba_data + rf_proba_data
                                    
                                    # Create plot
                                    fig = px.bar(
                                        all_proba_data, 
                                        x='Class', 
                                        y='Probability',
                                        color='Model',
                                        barmode='group',
                                        title='Prediction Probabilities by Model',
                                        height=400
                                    )
                                    fig.update_layout(yaxis_range=[0, 1])
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Disease information based on prediction
                                    most_likely_class = svm_pred if svm_probas[svm_pred] > rf_probas[rf_pred] else rf_pred
                                    
                                    st.subheader("Disease Information")
                                    
                                    if "Early_blight" in most_likely_class:
                                        st.info("**Early Blight** is caused by the fungus *Alternaria solani*. It typically appears as dark brown to black lesions with concentric rings, giving a 'target spot' appearance.")
                                        st.write("**Management**: Fungicides, crop rotation, removing infected plants, and maintaining good spacing for air circulation.")
                                    
                                    elif "Late_blight" in most_likely_class:
                                        st.warning("**Late Blight** is caused by the water mold *Phytophthora infestans* (the cause of the Irish Potato Famine). It appears as dark, water-soaked lesions that quickly enlarge and a white, fuzzy growth on the underside of leaves in moist conditions.")
                                        st.write("**Management**: Fungicides, destroying infected plants immediately, planting resistant varieties, and avoiding overhead irrigation.")
                                    
                                    elif "healthy" in most_likely_class:
                                        st.success("The leaf appears to be **healthy**. Continue with good agricultural practices to maintain plant health.")
                                        st.write("**Best Practices**: Proper spacing, adequate watering, balanced fertilization, and regular monitoring for early signs of disease.")
                                    
                                    else:
                                        st.write(f"Class: {most_likely_class} - No specific information available.")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Footer
        st.markdown("---")
        st.write("Developed for plant disease detection using machine learning.")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.code(traceback.format_exc())

# Add missing imports
import pandas as pd
import traceback

if __name__ == "__main__":
    main()
