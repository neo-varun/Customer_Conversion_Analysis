# Customer Conversion Analysis App

## Overview
This project is a **Customer Conversion Analysis App** built using **Streamlit**. It allows businesses to **analyze, predict, and visualize customer behavior** based on **browsing patterns, product interactions, and conversion data**.

The application provides interactive dashboards for **classification** (predicting conversions), **regression** (predicting prices), **clustering** (customer segmentation), and **exploratory data analysis** (visualizing trends and patterns).

## Features
- **Classification Models** – Predict whether a user will convert (make a purchase)
- **Regression Models** – Predict optimal product pricing
- **Customer Clustering** – Segment customers based on browsing and purchasing behavior
- **Interactive Interface** – Easy-to-use inputs for real-time predictions
- **Model Performance Metrics** – Compare different algorithms with key performance indicators
- **Exploratory Data Analysis** – Visualize data patterns with interactive charts and graphs
- **Multi-dimensional Visualizations** – Analyze relationships between variables with various plot types

## Prerequisites
Ensure you have **Python 3.x** installed on your system.

## Installation & Setup

### Create a Virtual Environment (Recommended)
It is recommended to create a virtual environment to manage dependencies:
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows:**  
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**  
  ```bash
  source venv/bin/activate
  ```

### Install Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
Run the app.py file:
```bash
streamlit run app.py
```

## How the Program Works

### Application Initialization
- When the application is launched, it creates necessary directories for artifacts and data if they don't exist.

### Classification Models
- Train various classification models including **Logistic Regression**, **Decision Trees**, **Random Forest**, and **XGBoost**.
- Models are trained with both **SMOTE oversampling** and **random undersampling** to handle class imbalance.
- Input customer features to predict the likelihood of conversion (purchase).

### Regression Models
- Train models like **Linear Regression**, **Ridge**, **Lasso**, and **Gradient Boosting** to predict optimal product pricing.
- Compare models using **RMSE**, **MAE**, and **R²** metrics.

### Clustering Models
- Segment customers using **KMeans** and **Agglomerative clustering**.
- Analyze cluster characteristics including browsing behavior, product preferences, and conversion rates.
- Generate customer personas based on identified segments.

### Data Processing Pipeline
- **Feature Engineering**: Extract session metrics, clickstream patterns, and behavioral metrics.
- **Data Preprocessing**: Handle missing values, scale numerical features, and encode categorical variables.

### Exploratory Data Analysis (EDA)
- **Interactive Visualizations**: Five different visualization types to explore the raw data
- **Line Plots**: Analyze price trends across product categories and seasons
- **Bar Charts**: Compare session distributions across different countries
- **Histograms**: Examine price distributions for each product category
- **Pie Charts**: View product category distribution with integrated price analysis
- **Correlation Heatmaps**: Discover relationships between different features

## Usage Guide

### Train Models
- Select a tab (**Classification**, **Regression**, or **Clustering**).
- Click the **"Train"** button to process data and train the corresponding models.
- View performance metrics to compare different algorithms.

### Make Predictions
- After training, navigate to the **User Inputs** tab.
- Fill in the form with customer and product information.
- Click **"Predict"** to see the results:
  - Classification: Conversion probability
  - Regression: Predicted optimal price

### Analyze Customer Segments
- After training clustering models, explore the **Cluster Analysis** section.
- View detailed statistics and characteristics of each customer segment.

### Explore Data Visualizations
- Navigate to the **EDA** tab to access various data visualizations
- View price trends by product category across different months
- Analyze the distribution of sessions by country
- Examine price distributions for each product category
- Explore product category distribution and pricing analysis
- Discover correlations between different features using interactive heatmaps

## Technologies Used
- **Python**
- **Streamlit** (Frontend & UI)
- **Scikit-learn** (Machine Learning)
- **XGBoost** (Gradient Boosting)
- **Imbalanced-learn** (Class Imbalance Handling)
- **Pandas & NumPy** (Data Manipulation)
- **Matplotlib & Seaborn** (Advanced Data Visualization)

## License
This project is licensed under the **MIT License**.

## Author
Developed by **Varun**. Feel free to connect with me on:
- **Email:** darklususnaturae@gmail.com
