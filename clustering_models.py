import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

class ClusteringModels:

    def __init__(self):
        self.df = pd.read_csv("data/clean_data.csv")
        
        # Define feature groups
        self.browsing_features = [
            'session_length', 'unique_pages', 'bounce', 'exit_rate'
        ]
        
        self.product_features = [
            'page1_main_category', 'colour', 'price_2'
        ] 
        
        self.interaction_features = [
            'click_sequence', 'revisit', 'order'
        ]
        
        # Use all three feature groups
        self.selected_features = self.browsing_features + self.product_features + self.interaction_features
        
        # Filter X to include only selected features that are in the dataframe
        self.X = self.df[[col for col in self.selected_features if col in self.df.columns]]
        
        # Handle categorical features if needed
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using silhouette score"""
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(self.X_scaled)
            
            score = silhouette_score(self.X_scaled, labels)
            silhouette_scores.append(score)
            
        # Find the optimal number of clusters
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        return optimal_clusters, silhouette_scores

    def analyze_clusters(self, model, labels):
        """Analyze cluster characteristics to create user personas"""
        # Add cluster labels to original DataFrame
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = labels
        
        # Analyze each cluster
        cluster_analysis = {}
        
        for cluster_id in range(len(set(labels))):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Calculate statistics for this cluster
            cluster_stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_with_clusters) * 100
            }
            
            # Analyze browsing behavior
            for feature in self.browsing_features:
                if feature in cluster_data.columns:
                    cluster_stats[f'avg_{feature}'] = cluster_data[feature].mean()
            
            # Analyze product preferences
            for feature in self.product_features:
                if feature in cluster_data.columns:
                    if feature == 'page1_main_category':
                        # Get most common category
                        most_common = cluster_data[feature].mode()[0]
                        category_map = {1: 'trousers', 2: 'skirts', 3: 'blouses', 4: 'sale'}
                        cluster_stats['preferred_category'] = category_map.get(most_common, str(most_common))
                    elif feature == 'colour':
                        # Get most common color
                        most_common = cluster_data[feature].mode()[0]
                        color_map = {1: 'beige', 2: 'black', 3: 'blue', 4: 'brown', 5: 'burgundy', 
                                    6: 'gray', 7: 'green', 8: 'navy blue', 9: 'many colors', 
                                    10: 'olive', 11: 'pink', 12: 'red', 13: 'violet', 14: 'white'}
                        cluster_stats['preferred_color'] = color_map.get(most_common, str(most_common))
                    else:
                        cluster_stats[f'avg_{feature}'] = cluster_data[feature].mean()
            
            # Analyze interaction patterns
            for feature in self.interaction_features:
                if feature in cluster_data.columns:
                    cluster_stats[f'avg_{feature}'] = cluster_data[feature].mean()
            
            # Add conversion rate
            if 'conversion' in cluster_data.columns:
                cluster_stats['conversion_rate'] = cluster_data['conversion'].mean() * 100
            
            # Use 1-based indexing for cluster IDs (cluster_id + 1)
            cluster_analysis[cluster_id + 1] = cluster_stats
        
        return cluster_analysis

    def clustering_models(self):
        # Find optimal number of clusters
        optimal_clusters, silhouette_scores = self.find_optimal_clusters()
        
        models = {
            "KMeans": KMeans(n_clusters=optimal_clusters, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=optimal_clusters)
        }

        results = {}
        cluster_analyses = {}
        
        # Create artifacts directories if they don't exist
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")
        if not os.path.exists("artifacts/clustering"):
            os.makedirs("artifacts/clustering")

        for name, model in models.items():
            labels = model.fit_predict(self.X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 1:
                metrics = {
                    "Silhouette Score": silhouette_score(self.X_scaled, labels),
                    "Davies-Bouldin Index": davies_bouldin_score(self.X_scaled, labels),
                    "Number of Clusters": n_clusters
                }
            else:
                metrics = {
                    "Silhouette Score": None,
                    "Davies-Bouldin Index": None,
                    "Number of Clusters": n_clusters
                }

            # Analyze clusters
            cluster_analysis = self.analyze_clusters(model, labels)
            cluster_analyses[name] = cluster_analysis
            
            results[name] = metrics
            
            # Save each model in the clustering folder with its name
            model_path = f"artifacts/clustering/{name}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        
        # Save cluster analyses
        with open("artifacts/clustering/cluster_analyses.pkl", "wb") as f:
            pickle.dump(cluster_analyses, f)

        return results, cluster_analyses