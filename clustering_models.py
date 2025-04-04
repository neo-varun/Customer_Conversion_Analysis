import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

X = pd.read_csv("data/clean_data.csv")

models = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Agglomerative": AgglomerativeClustering(n_clusters=3)
}

for name, model in models.items():
    print(f"\n{name} Results:")
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        print(f"Silhouette Score      : {sil_score:.4f}")
        print(f"Davies-Bouldin Index  : {db_score:.4f}")
    else:
        print("Not enough clusters to evaluate (only one found).")