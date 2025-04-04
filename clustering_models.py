import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

class ClusteringModels:

    def __init__(self):
        self.df = pd.read_csv("data/clean_data.csv")
        self.X = self.df.drop(columns=['conversion'])

    def clustering_models(self):
        models = {
            "KMeans": KMeans(n_clusters=3, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=3)
        }

        results = {}

        for name, model in models.items():
            labels = model.fit_predict(self.X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 1:
                metrics = {
                    "Silhouette Score": silhouette_score(self.X, labels),
                    "Davies-Bouldin Index": davies_bouldin_score(self.X, labels),
                    "Number of Clusters": n_clusters
                }
            else:
                metrics = {
                    "Silhouette Score": None,
                    "Davies-Bouldin Index": None,
                    "Number of Clusters": n_clusters
                }

            results[name] = metrics

        return results