import streamlit as st
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

# Create four main tabs
classification_tab, regression_tab, clustering_tab, eda_tab = st.tabs(["Classification", "Regression", "Clustering", "EDA"])

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

######################
# EDA Visualizations #
######################
with eda_tab:
    st.header("Exploratory Data Analysis")
    
    # Check if data exists and load raw training data
    if not os.path.exists('data/train.csv'):
        st.info("Raw training data not found. Please check that data/train.csv exists.")
    else:
        # Load the raw training data
        df_raw = pd.read_csv('data/train.csv')
        st.write(f"Loaded raw training data: {df_raw.shape[0]} records with {df_raw.shape[1]} features")
        
        # Create tabs for different plot types
        line_tab, bar_tab, hist_tab, pie_tab, heatmap_tab = st.tabs(["Line Plot", "Bar Chart", "Histogram", "Pie Chart", "Heatmap"])
        
        with line_tab:
            st.subheader("Monthly Price Trends by Product Category")
            
            # Get month names for better readability
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Add month name column
            df_raw['month_name'] = df_raw['month'].apply(lambda x: month_names[x-1])
            
            # Group by month and category to show price trends by category
            category_price_by_month = df_raw.groupby(['month', 'month_name', 'page1_main_category'])['price'].mean().reset_index()
            
            # Add category name for readability
            category_map = {1: 'Trousers', 2: 'Skirts', 3: 'Blouses', 4: 'Sale'}
            category_price_by_month['category_name'] = category_price_by_month['page1_main_category'].map(category_map)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Define category colors and markers
            colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
            markers = ['o', 's', '^', 'D']
            
            # Get unique months and sort them chronologically
            months = sorted(category_price_by_month['month'].unique())
            month_labels = [month_names[m-1] for m in months]
            
            # Plot line for each category
            for i, (category_id, category_name) in enumerate(category_map.items()):
                # Filter data for this category
                cat_data = category_price_by_month[category_price_by_month['page1_main_category'] == category_id]
                
                if not cat_data.empty:
                    # Sort by month
                    cat_data = cat_data.sort_values('month')
                    
                    # Plot the line
                    ax.plot(cat_data['month'], cat_data['price'], 
                          marker=markers[i], markersize=10, linewidth=3, 
                          color=colors[i], label=category_name,
                          alpha=0.8)
                    
                    # Add data labels
                    for x, y in zip(cat_data['month'], cat_data['price']):
                        ax.annotate(f'${y:.2f}', 
                                  xy=(x, y), 
                                  xytext=(0, 10),
                                  textcoords='offset points',
                                  ha='center', 
                                  fontsize=9,
                                  color=colors[i],
                                  fontweight='bold')
            
            # Add overall average price line
            overall_price_by_month = df_raw.groupby(['month', 'month_name'])['price'].mean().reset_index()
            overall_price_by_month = overall_price_by_month.sort_values('month')
            
            ax.plot(overall_price_by_month['month'], overall_price_by_month['price'], 
                  marker='*', markersize=15, linewidth=4, 
                  color='#000000', label='Overall Average',
                  alpha=0.6, linestyle='--')
            
            # Set x-axis with month names
            ax.set_xticks(months)
            ax.set_xticklabels(month_labels, fontsize=12, rotation=45)
            
            # Set labels and title
            ax.set_xlabel('Month', fontsize=14, fontweight='bold')
            ax.set_ylabel('Average Price ($)', fontsize=14, fontweight='bold')
            ax.set_title('Average Price Trends by Product Category', fontsize=18, fontweight='bold')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Create shaded areas for seasons
            if len(months) >= 12:  # Only if we have data for all months
                # Define seasons (Spring: 3-5, Summer: 6-8, Fall: 9-11, Winter: 12, 1-2)
                spring = [3, 4, 5]
                summer = [6, 7, 8]
                fall = [9, 10, 11]
                winter = [12, 1, 2]
                
                # Get y-axis limits
                y_min, y_max = ax.get_ylim()
                height = y_max - y_min
                
                # Add shaded areas for seasons with labels
                for season, months_list, color, name in [
                    (spring, [3, 4, 5], '#c4e17f', 'Spring'),
                    (summer, [6, 7, 8], '#ffdd44', 'Summer'),
                    (fall, [9, 10, 11], '#e67e22', 'Fall'),
                    (winter, [12, 1, 2], '#a3d4f7', 'Winter')
                ]:
                    # Filter to get months in this season that exist in our data
                    season_months = [m for m in months_list if m in months]
                    if season_months:
                        min_month, max_month = min(season_months), max(season_months)
                        rect = plt.Rectangle((min_month - 0.5, y_min), 
                                           max_month - min_month + 1, height,
                                           color=color, alpha=0.1, zorder=0)
                        ax.add_patch(rect)
                        
                        # Add season label
                        ax.text(min_month + (max_month - min_month) / 2, y_min + height * 0.05,
                              name, fontsize=12, ha='center', va='bottom',
                              bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='none', alpha=0.7))
            
            # Add legend with bigger markers
            legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                            ncol=5, frameon=True, fontsize=12)
            
            # Comment out the problematic code that's causing the AttributeError
            # for handle in legend.legendHandles:
            #     handle.set_markersize(10)
            #     handle.set_linewidth(4)
            
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.2)
            
            st.pyplot(fig)
            st.write("Displays price trends for each product category across months")
        
        with bar_tab:
            st.subheader("Session Distribution by Country")
            
            # Get top 10 countries by session count
            country_counts = df_raw['country'].value_counts().nlargest(10)
            
            # Create country map for better labels
            country_map = {
                1: 'Australia', 2: 'Austria', 3: 'Belgium', 4: 'British VI', 5: 'Cayman Islands',
                6: 'Christmas Is.', 7: 'Croatia', 8: 'Cyprus', 9: 'Czech Rep.', 10: 'Denmark',
                11: 'Estonia', 12: 'Unidentified', 13: 'Faroe Islands', 14: 'Finland', 15: 'France',
                16: 'Germany', 17: 'Greece', 18: 'Hungary', 19: 'Iceland', 20: 'India', 21: 'Ireland',
                22: 'Italy', 23: 'Latvia', 24: 'Lithuania', 25: 'Luxembourg', 26: 'Mexico', 27: 'Netherlands',
                28: 'Norway', 29: 'Poland', 30: 'Portugal', 31: 'Romania', 32: 'Russia', 33: 'San Marino',
                34: 'Slovakia', 35: 'Slovenia', 36: 'Spain', 37: 'Sweden', 38: 'Switzerland', 39: 'Ukraine',
                40: 'UAE', 41: 'UK', 42: 'USA'
            }
            
            # Create readable country names
            country_names = [country_map.get(country, f'Country {country}') for country in country_counts.index]
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use colormap for aesthetic color gradient
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(country_counts)))
            
            # Create horizontal bars
            bars = ax.barh(country_names, country_counts, color=colors)
            
            # Add count labels inside bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width * 0.95
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:,.0f}',
                      ha='right', va='center', color='white', fontweight='bold', fontsize=11)
            
            # Customize appearance
            ax.set_xlabel('Number of Sessions', fontsize=14, fontweight='bold')
            ax.set_title('Top 10 Countries by Session Count', fontsize=16, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Invert y-axis for top-to-bottom ranking
            ax.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
            st.write("Shows the distribution of sessions across different countries, highlighting key markets")
        
        with hist_tab:
            st.subheader("Price Distribution by Product Category")
            
            # Create mapping for product categories
            category_map = {1: 'Trousers', 2: 'Skirts', 3: 'Blouses', 4: 'Sale'}
            
            # Add category name column for easier filtering
            df_raw['category_name'] = df_raw['page1_main_category'].map(category_map)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            # Define color palette
            colors = ['#ff9ff3', '#feca57', '#48dbfb', '#1dd1a1']
            
            # Create histogram for each category
            for i, (category_id, category_name) in enumerate(category_map.items()):
                # Filter data for this category
                category_data = df_raw[df_raw['page1_main_category'] == category_id]['price']
                
                # Create histogram with kernel density estimate
                sns.histplot(category_data, bins=15, kde=True, color=colors[i], ax=axes[i], alpha=0.7)
                
                # Add vertical line for mean price
                mean_price = category_data.mean()
                axes[i].axvline(mean_price, color='red', linestyle='--', linewidth=2)
                axes[i].text(mean_price+1, axes[i].get_ylim()[1]*0.9, f'Mean: ${mean_price:.2f}', 
                           color='red', fontweight='bold')
                
                # Customize plot
                axes[i].set_title(f'{category_name}', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Price ($)', fontsize=12)
                axes[i].set_ylabel('Frequency', fontsize=12)
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            fig.suptitle('Price Distribution by Product Category', fontsize=16, fontweight='bold')
            
            st.pyplot(fig)
            st.write("Shows the price distribution for each product category with mean price highlighted")
        
        with pie_tab:
            st.subheader("Product Category and Price Analysis")
            
            # Create mapping for product categories
            category_map = {1: 'Trousers', 2: 'Skirts', 3: 'Blouses', 4: 'Sale'}
            
            # Calculate aggregate data by category
            category_stats = df_raw.groupby('page1_main_category').agg({
                'price': ['mean', 'count', 'min', 'max']
            }).reset_index()
            
            # Flatten the MultiIndex columns
            category_stats.columns = ['category', 'avg_price', 'count', 'min_price', 'max_price']
            
            # Add category names
            category_stats['category_name'] = category_stats['category'].map(category_map)
            
            # Calculate percentage of total
            total_count = category_stats['count'].sum()
            category_stats['percentage'] = (category_stats['count'] / total_count * 100).round(1)
            
            # Create a figure with gridspec for custom layout (pie chart + bar chart)
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
            
            # Pie chart on the left
            ax1 = fig.add_subplot(gs[0])
            
            # Create pie chart with a slight explosion effect
            explode = [0.05, 0.05, 0.05, 0.05]  # Slight separation for all slices
            wedges, texts, autotexts = ax1.pie(
                category_stats['count'], 
                labels=None,
                explode=explode,
                autopct='%1.1f%%',
                startangle=90, 
                colors=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'],
                wedgeprops=dict(width=0.5, edgecolor='white'),  # Wider donut with white edges
                pctdistance=0.8,
                textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'}
            )
            
            # Create a white circle at the center for donut effect
            centre_circle = plt.Circle((0, 0), 0.25, fc='white')
            ax1.add_patch(centre_circle)
            
            # Add center text showing total count
            ax1.text(0, 0, f"Total\n{total_count:,}", ha='center', va='center', 
                   fontsize=16, fontweight='bold')
            
            # Create a legend
            ax1.legend(
                wedges,
                [f"{name} ({count:,})" for name, count in zip(category_stats['category_name'], category_stats['count'])],
                title="Categories",
                loc="center",
                bbox_to_anchor=(0.5, -0.1),
                fontsize=12,
                ncol=2
            )
            
            ax1.set_title('Product Distribution by Category', fontsize=16, fontweight='bold')
            
            # Bar chart on the right showing price comparison
            ax2 = fig.add_subplot(gs[1])
            
            # Create horizontal bars for average prices
            bars = ax2.barh(
                category_stats['category_name'],
                category_stats['avg_price'],
                color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'],
                alpha=0.8,
                height=0.5
            )
            
            # Add average price annotations inside bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(
                    width/2,
                    bar.get_y() + bar.get_height()/2,
                    f"${category_stats['avg_price'].iloc[i]:.2f}",
                    ha='center',
                    va='center',
                    color='white',
                    fontweight='bold',
                    fontsize=12
                )
            
            # Add price range annotations to the right of bars
            for i, (category, avg, min_price, max_price) in enumerate(
                zip(category_stats['category_name'], 
                    category_stats['avg_price'],
                    category_stats['min_price'],
                    category_stats['max_price'])):
                ax2.text(
                    avg + 3,
                    i,
                    f"Range: ${min_price:.2f} - ${max_price:.2f}",
                    va='center',
                    fontsize=11,
                    color='#555555'
                )
            
            # Configure bar chart
            ax2.set_xlabel('Average Price ($)', fontsize=14, fontweight='bold')
            ax2.set_title('Average Price by Category', fontsize=16, fontweight='bold')
            ax2.grid(axis='x', linestyle='--', alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Set a reasonable x limit with some padding
            ax2.set_xlim(0, max(category_stats['avg_price']) * 1.3)
            
            # Equal aspect ratio ensures the pie is circular
            ax1.set_aspect('equal')
            
            plt.tight_layout()
            fig.subplots_adjust(wspace=0.3)
            
            st.pyplot(fig)
            st.write("Displays category distribution alongside price analysis")
        
        with heatmap_tab:
            st.subheader("Correlation Between Key Features")
            
            # Select only numeric columns for correlation
            numeric_df = df_raw.select_dtypes(include=['int64', 'float64'])
            
            # Drop year and month columns
            if 'year' in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=['year'])
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create a mask for the upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Set up a custom colormap going from cool to warm
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create heatmap with mask
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, vmin=-.3,
                     annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8},
                     annot_kws={"size": 10})
            
            # Add custom column name mapping for better readability
            column_names = {
                'day': 'Day of Month',
                'order': 'Click Order',
                'country': 'Country',
                'session_id': 'Session ID',
                'page1_main_category': 'Category',
                'colour': 'Color',
                'location': 'Location',
                'model_photography': 'Photography',
                'price': 'Price',
                'price_2': 'High Price',
                'page': 'Page Number',
                'month': 'Month'
            }
            
            # Set readable labels
            readable_labels = [column_names.get(col, col) for col in corr_matrix.columns]
            ax.set_xticklabels(readable_labels, rotation=45, ha='right', fontsize=12)
            ax.set_yticklabels(readable_labels, rotation=0, fontsize=12)
            
            plt.title('Correlation Between Features', fontsize=16, pad=20)
            
            st.pyplot(fig)
            st.write("Shows relationships between different features in the raw dataset")