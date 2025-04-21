import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.spatial.distance import euclidean
from kneed import KneeLocator
from minisom import MiniSom

# 1. Pré-processamento: Leitura e remoção de outliers
df = pd.read_csv('Iris.csv')

# Remover coluna ID se existir
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# Boxplot para verificar outliers
features = df.columns[:-1]
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Normalização
X = df.iloc[:, :-1].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 2. Encontrando K ideal com Silhouette e Elbow
sil_scores = []
wcss = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42)
    pred = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, pred)
    sil_scores.append(sil)
    wcss.append(model.inertia_)
    print(f"K={k} | Silhouette={sil:.3f} | WCSS={model.inertia_:.2f}")

# Elbow
kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
optimal_k = kl.elbow
print(f"Número ideal de clusters (elbow): {optimal_k}")

plt.figure()
plt.plot(range(2, 11), wcss, 'bo-')
plt.axvline(x=optimal_k, color='r', linestyle='--', label='Elbow')
plt.xlabel("Número de Clusters")
plt.ylabel("Soma dos Erros (WCSS)")
plt.title("Método do Cotovelo")
plt.legend()
plt.show()

# 3. Executar KMeans com k ideal
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 4. Hiperparâmetros e métricas
# Distância Euclidiana
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Silhouette Score já calculado
# Implementando Dunn Index
def dunn_index(data, labels):
    unique_clusters = np.unique(labels)
    intra_dists = []
    inter_dists = []

    for c in unique_clusters:
        points = data[labels == c]
        if len(points) > 1:
            intra = max([euclidean(p1, p2) for i, p1 in enumerate(points) for p2 in points[i+1:]])
            intra_dists.append(intra)
        else:
            intra_dists.append(0)

    for i in range(len(unique_clusters)):
        for j in range(i+1, len(unique_clusters)):
            cluster_i = data[labels == unique_clusters[i]]
            cluster_j = data[labels == unique_clusters[j]]
            inter = min([euclidean(p1, p2) for p1 in cluster_i for p2 in cluster_j])
            inter_dists.append(inter)

    return min(inter_dists) / max(intra_dists)

dunn = dunn_index(X_scaled, kmeans_labels)
print(f"Dunn Index: {dunn:.3f}")

# 5. DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 6. SOM
som = MiniSom(x=3, y=1, input_len=4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, 100)

# Mapeando as posições dos neurônios vencedores
som_labels = np.array([np.ravel_multi_index(som.winner(x), (3, 1)) for x in X_scaled])

# 7. Visualizando erros (com base no rótulo original)
from sklearn.preprocessing import LabelEncoder

y_true = LabelEncoder().fit_transform(df['Species'])

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y_true, palette='deep')
plt.title("Rótulos Originais")

plt.subplot(1, 3, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=kmeans_labels, palette='deep')
plt.title("Agrupamento KMeans")

plt.subplot(1, 3, 3)
incorrect = y_true != kmeans_labels
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=incorrect, palette=['green', 'red'])
plt.title("Instâncias Agrupadas Incorretamente")
plt.legend(title='Erro')
plt.tight_layout()
plt.show()

# 8. Relatório: será montado com os resultados que você coletar rodando este código
