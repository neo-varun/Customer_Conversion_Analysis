import streamlit as st
import pandas as pd
import os
import pickle
from classification_models import ClassificationModels
from regression_models import RegressionModels
from clustering_models import ClusteringModels
from feature_engineering import FeatureEngineering
from data_preprocessing import DataPreprocessor

st.title("Model Training & Performance Dashboard")

def process_data():
    """Process data through feature engineering and preprocessing pipeline"""
    # Check if data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Step 1: Feature Engineering
    fe = FeatureEngineering()
    fe.process_and_save('data/train.csv', 'data/feature_data.csv')
    fe.save_preprocessor()
    
    # Step 2: Data Preprocessing
    dp = DataPreprocessor()
    dp.preprocess_file('data/feature_data.csv', 'data/clean_data.csv')
    dp.save_preprocessor()

def create_input_form(key_prefix, disable_price=False):
    """Creates input fields using your mapping categories.
       disable_price: If True, the price field will be disabled (for regression prediction)
    """
    col1, col2, col3 = st.columns(3)
    
    # Column 1 – Basic Information
    with col1:
        # 3. DAY: day number of the month
        day = st.number_input("DAY (day number of the month)", min_value=1, max_value=31, value=1, key=f"{key_prefix}_day")
        # 4. ORDER: sequence of clicks during one session
        order = st.number_input("ORDER (sequence of clicks)", min_value=1, value=1, key=f"{key_prefix}_order")
        # 5. COUNTRY: use the mapping provided
        countries = [
            "1-Australia", "2-Austria", "3-Belgium", "4-British Virgin Islands", "5-Cayman Islands",
            "6-Christmas Island", "7-Croatia", "8-Cyprus", "9-Czech Republic", "10-Denmark",
            "11-Estonia", "12-unidentified", "13-Faroe Islands", "14-Finland", "15-France",
            "16-Germany", "17-Greece", "18-Hungary", "19-Iceland", "20-India", "21-Ireland",
            "22-Italy", "23-Latvia", "24-Lithuania", "25-Luxembourg", "26-Mexico", "27-Netherlands",
            "28-Norway", "29-Poland", "30-Portugal", "31-Romania", "32-Russia", "33-San Marino",
            "34-Slovakia", "35-Slovenia", "36-Spain", "37-Sweden", "38-Switzerland", "39-Ukraine",
            "40-United Arab Emirates", "41-United Kingdom", "42-USA", "43-biz (.biz)", "44-com (.com)",
            "45-int (.int)", "46-net (.net)", "47-org (*.org)"
        ]
        country_selection = st.selectbox("COUNTRY", options=countries, key=f"{key_prefix}_country")
        country = int(country_selection.split("-")[0])
        # 6. SESSION ID
        session_id = st.number_input("SESSION ID", value=12345, key=f"{key_prefix}_session")
    
    # Column 2 – Product Information
    with col2:
        # 7. PAGE 1 (MAIN CATEGORY)
        main_categories = ["1-trousers", "2-skirts", "3-blouses", "4-sale"]
        category_selection = st.selectbox("PAGE 1 (MAIN CATEGORY)", options=main_categories, key=f"{key_prefix}_category")
        category = int(category_selection.split("-")[0])
        # 8. PAGE 2 (CLOTHING MODEL)
        valid_models = ["A1", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A2", "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29", "A3", "A30", "A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39", "A4", "A40", "A41", "A42", "A43", "A5", "A6", "A7", "A8", "A9", "B1", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B19", "B2", "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28", "B29", "B3", "B30", "B31", "B32", "B33", "B34", "B4", "B5", "B6", "B7", "B8", "B9", "C1", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C2", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C3", "C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C4", "C40", "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C5", "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C59", "C6", "C7", "C8", "C9", "P1", "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19", "P2", "P20", "P21", "P23", "P24", "P25", "P26", "P27", "P29", "P3", "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37", "P38", "P39", "P4", "P40", "P41", "P42", "P43", "P44", "P45", "P46", "P47", "P48", "P49", "P5", "P50", "P51", "P52", "P53", "P55", "P56", "P57", "P58", "P59", "P6", "P60", "P61", "P62", "P63", "P64", "P65", "P66", "P67", "P68", "P69", "P7", "P70", "P71", "P72", "P73", "P74", "P75", "P76", "P77", "P78", "P8", "P80", "P81", "P82", "P9"]
        clothing_model = st.selectbox("CLOTHING MODEL", options=valid_models, key=f"{key_prefix}_model")
        # 9. COLOUR
        colours = [
            "1-beige", "2-black", "3-blue", "4-brown", "5-burgundy", "6-gray", "7-green",
            "8-navy blue", "9-of many colors", "10-olive", "11-pink", "12-red", "13-violet", "14-white"
        ]
        color_selection = st.selectbox("COLOUR", options=colours, key=f"{key_prefix}_color")
        color = int(color_selection.split("-")[0])
        # 10. LOCATION
        locations = [
            "1-top left", "2-top in the middle", "3-top right",
            "4-bottom left", "5-bottom in the middle", "6-bottom right"
        ]
        location_selection = st.selectbox("LOCATION", options=locations, key=f"{key_prefix}_location")
        location = int(location_selection.split("-")[0])
    
    # Column 3 – Additional Information
    with col3:
        # 11. MODEL PHOTOGRAPHY
        photography_options = ["1-en face", "2-profile"]
        photo_selection = st.selectbox("MODEL PHOTOGRAPHY", options=photography_options, key=f"{key_prefix}_photo")
        model_photography = int(photo_selection.split("-")[0])
        
        # 12. PRICE - Disabled for regression prediction
        if disable_price:
            price = None
            st.info("Price will be predicted")
        else:
            price = st.number_input("PRICE (USD)", min_value=0.0, value=50.0, step=0.01, key=f"{key_prefix}_price")
            
        # 13. PRICE 2 (higher than average?): mapped as 1-yes, 2-no
        price2_options = ["1-yes", "2-no"]
        price2_selection = st.selectbox("PRICE 2 (higher than average?)", options=price2_options, key=f"{key_prefix}_higher")
        price2 = int(price2_selection.split("-")[0])
        # 14. PAGE: page number within the e-store (1 to 5)
        page = st.number_input("PAGE", min_value=1, max_value=5, value=1, step=1, key=f"{key_prefix}_page")
    
    input_data = {
        "day": day,
        "order": order,
        "country": country,
        "session_id": session_id,
        "page1_main_category": category,
        "page2_clothing_model": clothing_model,
        "colour": color,
        "location": location,
        "model_photography": model_photography,
        "price_2": price2,
        "page": page
    }
    
    # Add price only if it's not disabled
    if not disable_price:
        input_data["price"] = price
        
    return input_data

def prepare_single_input_for_prediction(input_data, model_type='regression'):
    """Process a single input through the preprocessing pipeline"""
    # Create a DataFrame from the input
    input_df = pd.DataFrame([input_data])
    
    # Add price column with placeholder if missing (for regression prediction)
    if 'price' not in input_df.columns:
        input_df['price'] = 0.0  # Placeholder value
    
    # Load the feature engineering preprocessor
    with open("artifacts/preprocessor_feature.pkl", "rb") as f:
        fe_preprocessor = pickle.load(f)
    
    # Load the data cleaning preprocessor    
    with open("artifacts/preprocessor_clean.pkl", "rb") as f:
        clean_preprocessor = pickle.load(f)
    
    # Create instances and set their preprocessors
    fe = FeatureEngineering()
    fe.preprocessor = fe_preprocessor
    
    # Apply feature engineering
    processed_input = fe.process(input_df)
    
    # Apply data preprocessing
    dp = DataPreprocessor()
    dp.preprocessor = clean_preprocessor
    processed_input = dp.preprocess_data(processed_input, model_type)
    
    return processed_input

def predict_price(input_data, model_name="GradientBoosting"):
    """Predict price using the selected regression model"""
    # Process the input data
    processed_input = prepare_single_input_for_prediction(input_data, model_type='regression')
    
    # Load the selected regression model
    model_path = f"artifacts/regression/{model_name}_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Make prediction
    raw_prediction = model.predict(processed_input)[0]
    
    # Take absolute value to ensure non-negative price
    price_modulus = abs(raw_prediction)
    
    # Apply scaling factor to make predictions realistic
    scaling_factor = 100.0
    adjusted_prediction = price_modulus * scaling_factor
    
    # Ensure minimum reasonable price
    return max(1.0, adjusted_prediction)

def predict_conversion(input_data, model_name="RandomForest_SMOTE"):
    """Predict if a user will convert (buy) based on the input features"""
    # Copy the input data to avoid modifying the original
    input_data_copy = input_data.copy()
    
    # Process the input data - page column will be handled internally
    processed_input = prepare_single_input_for_prediction(input_data_copy, model_type='classification')
    
    # Load the selected classification model
    model_path = f"artifacts/classification/{model_name}_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Make prediction
    prediction_proba = model.predict_proba(processed_input)
    prediction = model.predict(processed_input)[0]
    conversion_probability = prediction_proba[0][1]  # Probability of class 1 (conversion)
    
    return prediction, conversion_probability

# Create three main tabs
classification_tab, regression_tab, clustering_tab = st.tabs(["Classification", "Regression", "Clustering"])

#########################
# Classification Models #
#########################
with classification_tab:
    st.header("Classification Models")
    # Create sub-tabs for Training and User Inputs
    train_tab, input_tab = st.tabs(["Training", "User Inputs"])
    
    with train_tab:
        st.subheader("Train & Evaluate Classification Models")
        if st.button("Train Classification Models", key="train_classification"):
            st.write("Processing data and training classification models... Please wait.")
            # Process data through the pipeline
            process_data()
            # Train models
            classifier = ClassificationModels()
            class_results = classifier.classification_models()
            st.session_state["classification_results"] = class_results
            st.success("Classification model training complete!")
        
        if "classification_results" in st.session_state:
            st.subheader("Classification Model Performance")
            results_df = pd.DataFrame.from_dict(st.session_state["classification_results"], orient="index")
            results_df.reset_index(inplace=True)
            results_df.rename(columns={'index': 'Model'}, inplace=True)
            # Separate results for SMOTE and Undersampling
            smote_df = results_df[results_df["Model"].str.contains("SMOTE")].reset_index(drop=True)
            undersample_df = results_df[results_df["Model"].str.contains("UnderSampling")].reset_index(drop=True)
            smote_df.index = smote_df.index + 1  # start index at 1
            undersample_df.index = undersample_df.index + 1
            
            st.write("### SMOTE Oversampling Results")
            st.table(smote_df)
            st.write("### Random Undersampling Results")
            st.table(undersample_df)
    
    with input_tab:
        if "classification_results" not in st.session_state:
            st.info("Please train the classification models first in the Training tab.")
        else:
            st.subheader("Conversion Prediction")
            
            # Model selection
            model_options = ["LogisticRegression_SMOTE", "DecisionTree_SMOTE", 
                             "RandomForest_SMOTE", "XGBoost_SMOTE"]
            selected_model = st.selectbox(
                "Select Classification Model",
                model_options,
                index=2,  # Default to RandomForest_SMOTE
                key="classification_model_selection"
            )
            
            # Get inputs
            classification_inputs = create_input_form("classification")
            
            # Add a button to make prediction
            if st.button("Predict Conversion", key="predict_conversion"):
                try:
                    # Make the prediction with selected model
                    prediction, probability = predict_conversion(classification_inputs, selected_model)
                    
                    # Display the result
                    if prediction == 1:
                        st.success(f"User is likely to purchase!")
                    else:
                        st.warning(f"User is unlikely to buy.")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

######################
# Regression Models  #
######################
with regression_tab:
    st.header("Regression Models")
    reg_train_tab, reg_predict_tab = st.tabs(["Training", "User Inputs"])
    
    with reg_train_tab:
        st.subheader("Train & Evaluate Regression Models")
        if st.button("Train Regression Models", key="train_regression"):
            st.write("Processing data and training regression models... Please wait.")
            # Process data through the pipeline
            process_data()
            # Train models
            regressor = RegressionModels()
            reg_results = regressor.regression_models()
            st.session_state["regression_results"] = reg_results
            st.success("Regression model training complete!")
    
        if "regression_results" in st.session_state:
            st.subheader("Regression Model Performance")
            results_df = pd.DataFrame.from_dict(st.session_state["regression_results"], orient="index")
            results_df.reset_index(inplace=True)
            results_df.rename(columns={'index': 'Model'}, inplace=True)
            results_df.index = results_df.index + 1
            st.table(results_df)

    with reg_predict_tab:
        if "regression_results" not in st.session_state:
            st.info("Please train the regression models first in the Training tab.")
        else:
            st.subheader("Price Prediction")
            
            # Model selection
            model_options = ["LinearRegression", "Ridge", "Lasso", "GradientBoosting"]
            selected_model = st.selectbox(
                "Select Regression Model",
                model_options,
                index=3,  # Default to GradientBoosting
                key="regression_model_selection"
            )
            
            # Get inputs without price field
            regression_inputs = create_input_form("regression", disable_price=True)
            
            # Add a button to make prediction
            if st.button("Predict Price", key="predict_price"):
                try:
                    # Make the prediction with selected model
                    predicted_price = predict_price(regression_inputs, selected_model)
                    
                    # Display the result
                    st.success(f"Predicted Price: ${predicted_price:.2f}")
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

######################
# Clustering Models  #
######################
with clustering_tab:
    st.header("Clustering Models")
    train_tab, input_tab = st.tabs(["Training", "User Inputs"])
    
    with train_tab:
        st.subheader("Train & Evaluate Clustering Models")
        if st.button("Train Clustering Models", key="train_clustering"):
            st.write("Processing data and training clustering models... Please wait.")
            # Process data through the pipeline
            process_data()
            # Train models
            clusterer = ClusteringModels()
            cluster_results, cluster_analyses = clusterer.clustering_models()
            st.session_state["clustering_results"] = cluster_results
            st.session_state["cluster_analyses"] = cluster_analyses
            st.success("Clustering model training complete!")
        
        if "clustering_results" in st.session_state:
            st.subheader("Clustering Model Performance")
            results_df = pd.DataFrame.from_dict(st.session_state["clustering_results"], orient="index")
            results_df.reset_index(inplace=True)
            results_df.rename(columns={'index': 'Model'}, inplace=True)
            results_df.index = results_df.index + 1
            st.table(results_df)
    
    with input_tab:
        if "clustering_results" not in st.session_state:
            st.info("Please train the clustering models first in the Training tab.")
        else:
            st.subheader("Cluster Analysis")
            
            if os.path.exists("artifacts/clustering/cluster_analyses.pkl"):
                # Load the cluster analyses
                with open("artifacts/clustering/cluster_analyses.pkl", "rb") as f:
                    cluster_analyses = pickle.load(f)
                
                # Create tabs for each clustering algorithm
                algorithm_tabs = st.tabs(list(cluster_analyses.keys()))
                
                for i, (algorithm, tab) in enumerate(zip(cluster_analyses.keys(), algorithm_tabs)):
                    with tab:
                        st.subheader(f"{algorithm} Clustering Results")
                        
                        # Display cluster statistics
                        st.write("### Cluster Statistics")
                        for cluster_id, stats in cluster_analyses[algorithm].items():
                            with st.expander(f"Cluster {cluster_id}"):
                                # Convert stats to a more readable format
                                display_stats = {
                                    "Size": f"{stats['size']} users ({stats['percentage']:.1f}%)",
                                    "Browsing Behavior": {
                                        "Avg Session Length": f"{stats.get('avg_session_length', 'N/A'):.2f}",
                                        "Unique Pages Viewed": f"{stats.get('avg_unique_pages', 'N/A'):.2f}",
                                        "Bounce Rate": f"{stats.get('avg_bounce', 'N/A'):.2%}",
                                        "Exit Rate": f"{stats.get('avg_exit_rate', 'N/A'):.2%}"
                                    },
                                    "Product Preferences": {
                                        "Preferred Category": stats.get('preferred_category', 'N/A'),
                                        "Preferred Color": stats.get('preferred_color', 'N/A'),
                                        "Price Sensitivity": "High" if stats.get('avg_price_2', 0) > 1.5 else "Low"
                                    },
                                    "Interaction Patterns": {
                                        "Revisit Rate": f"{stats.get('avg_revisit', 'N/A'):.2%}",
                                        "Avg Order Position": f"{stats.get('avg_order', 'N/A'):.2f}"
                                    },
                                    "Conversion Rate": f"{stats.get('conversion_rate', 'N/A'):.2%}"
                                }
                                
                                st.json(display_stats)
                        
            else:
                st.warning("Cluster analyses not found. Please retrain the clustering models.")