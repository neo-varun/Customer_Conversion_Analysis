import streamlit as st
import pandas as pd
from classification_models import ClassificationModels
from regression_models import RegressionModels
from clustering_models import ClusteringModels

# Streamlit App Title
st.title("Model Training & Performance Dashboard")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["Classification", "Regression", "Clustering"])

# ========================= CLASSIFICATION TAB =========================
with tab1:
    st.subheader("Train & Evaluate Classification Models")

    if st.button("Train Classification Models"):
        st.write("Training classification models... Please wait.")

        classifier = ClassificationModels()
        class_results = classifier.classification_models()

        st.session_state["classification_results"] = class_results
        st.success("Classification model training complete!")

    # Display Classification Metrics
    if "classification_results" in st.session_state:
        st.subheader("Classification Model Performance")

        # Convert stored results to DataFrame for a cleaner display
        results_df = pd.DataFrame.from_dict(st.session_state["classification_results"], orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={'index': 'Model'}, inplace=True)

        # Separate SMOTE and UnderSampling models
        smote_df = results_df[results_df["Model"].str.contains("SMOTE")].reset_index(drop=True)
        undersampling_df = results_df[results_df["Model"].str.contains("UnderSampling")].reset_index(drop=True)

        # Start indexing from 1
        smote_df.index = smote_df.index + 1
        undersampling_df.index = undersampling_df.index + 1

        # Display SMOTE results
        st.write("### SMOTE Oversampling Results")
        st.table(smote_df)

        # Display Undersampling results
        st.write("### Random Undersampling Results")
        st.table(undersampling_df)

# ========================= REGRESSION TAB =========================
with tab2:
    st.subheader("Train & Evaluate Regression Models")

    if st.button("Train Regression Models"):
        st.write("Training regression models... Please wait.")

        regressor = RegressionModels()
        reg_results = regressor.regression_models()

        st.session_state["regression_results"] = reg_results
        st.success("Regression model training complete!")

    # Display Regression Metrics
    if "regression_results" in st.session_state:
        st.subheader("Regression Model Performance")

        results_df = pd.DataFrame.from_dict(st.session_state["regression_results"], orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={'index': 'Model'}, inplace=True)

        results_df.index = results_df.index + 1  # Start indexing from 1
        st.table(results_df)  # Display as a nicely formatted table

# ========================= CLUSTERING TAB =========================
with tab3:
    st.subheader("Train & Evaluate Clustering Models")

    if st.button("Train Clustering Models"):
        st.write("Training clustering models... Please wait.")

        clusterer = ClusteringModels()
        cluster_results = clusterer.clustering_models()

        st.session_state["clustering_results"] = cluster_results
        st.success("Clustering model training complete!")

    # Display Clustering Metrics
    if "clustering_results" in st.session_state:
        st.subheader("Clustering Model Performance")

        results_df = pd.DataFrame.from_dict(st.session_state["clustering_results"], orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={'index': 'Model'}, inplace=True)

        results_df.index = results_df.index + 1  # Start indexing from 1
        st.table(results_df)  # Display as a nicely formatted table