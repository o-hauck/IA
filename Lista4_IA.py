import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from sklearn.impute import SimpleImputer, KNNImputer

# Carregando a base Titanic
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Selecionando apenas colunas relevantes
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
df.dropna(subset=['Embarked'], inplace=True)

# Transformação de variáveis categóricas
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Separando features e labels
X = df.drop(columns=['Survived'])
y = df['Survived']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ===========================
# IMPUTAÇÃO DE DADOS AUSENTES
# ===========================

# Imputação usando Média
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)  # Garante consistência na imputação

# ===========================
# OTIMIZAÇÃO DE HIPERPARÂMETROS
# ===========================

# Definição dos espaços de busca
param_grid_rf = {'n_estimators': (10, 200), 'max_depth': (1, 20)}
param_grid_dt = {'max_depth': (1, 20), 'min_samples_split': (2, 10)}

# Aplicando BayesSearchCV com validação cruzada
rf_search = BayesSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, n_iter=30, cv=5, n_jobs=-1)
dt_search = BayesSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, n_iter=30, cv=5, n_jobs=-1)

rf_search.fit(X_train_imputed, y_train)
dt_search.fit(X_train_imputed, y_train)

# Selecionando os melhores modelos
rf_best = rf_search.best_estimator_
dt_best = dt_search.best_estimator_

# ===========================
# AVALIAÇÃO DOS MODELOS
# ===========================

def avaliar_modelo(nome, modelo, X_teste, y_teste):
    y_pred = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred)
    precisao = precision_score(y_teste, y_pred)
    recall = recall_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred)
    
    return {
        "Modelo": nome,
        "Acurácia": acuracia,
        "Precisão": precisao,
        "Recall": recall,
        "F1-Score": f1,
        "Matriz de Confusão": confusion_matrix(y_teste, y_pred)
    }

resultados = []
resultados.append(avaliar_modelo("Random Forest", rf_best, X_test_imputed, y_test))
resultados.append(avaliar_modelo("Árvore de Decisão", dt_best, X_test_imputed, y_test))

# ===========================
# EXIBIR ATRIBUTOS MAIS RELEVANTES
# ===========================

# Para o Random Forest
print("Atributos mais relevantes (Random Forest):")
for feature, importance in zip(X.columns, rf_best.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Para a Árvore de Decisão
print("\nAtributos mais relevantes (Árvore de Decisão):")
for feature, importance in zip(X.columns, dt_best.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# ===========================
# MÉTODOS DE BALANCEAMENTO
# ===========================

metodos_balanceamento = {
    "SMOTE": SMOTE(),
    "TomekLinks": TomekLinks(),
    "RandomUnderSampler": RandomUnderSampler(),
    "ADASYN": ADASYN()
}

for nome, metodo in metodos_balanceamento.items():
    X_res, y_res = metodo.fit_resample(X_train_imputed, y_train)

    # Treina o modelo otimizado com os dados balanceados
    modelo_balanceado = RandomForestClassifier(random_state=42)
    modelo_balanceado.fit(X_res, y_res)

    # Avaliação usando o mesmo conjunto de teste
    resultados.append(avaliar_modelo(f"Random Forest ({nome})", modelo_balanceado, X_test_imputed, y_test))

# ===========================
# COMPARAÇÃO DE IMPUTAÇÃO
# ===========================

metodos_imputacao = {
    "Média": SimpleImputer(strategy='mean'),
    "Moda": SimpleImputer(strategy='most_frequent'),
    "KNN": KNNImputer(n_neighbors=3)
}

for nome, imputer in metodos_imputacao.items():
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train_imp, y_train)

    resultados.append(avaliar_modelo(f"Random Forest ({nome})", modelo, X_test_imp, y_test))

# Convertendo os resultados para DataFrame
df_resultados = pd.DataFrame(resultados)

# ===========================
# GRÁFICO DE COMPARAÇÃO
# ===========================

# Gráfico de barras para cada métrica
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Acurácia
axes[0, 0].barh(df_resultados["Modelo"], df_resultados["Acurácia"], color="skyblue")
axes[0, 0].set_title("Acurácia")

# Precisão
axes[0, 1].barh(df_resultados["Modelo"], df_resultados["Precisão"], color="lightgreen")
axes[0, 1].set_title("Precisão")

# Recall
axes[1, 0].barh(df_resultados["Modelo"], df_resultados["Recall"], color="lightcoral")
axes[1, 0].set_title("Recall")

# F1-Score
axes[1, 1].barh(df_resultados["Modelo"], df_resultados["F1-Score"], color="lightyellow")
axes[1, 1].set_title("F1-Score")

plt.tight_layout()
plt.show()

# ===========================
# VISUALIZAÇÃO DAS MATRIZES DE CONFUSÃO
# ===========================

# Função para plotar a matriz de confusão
def plot_matriz_confusao(matriz, nome):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=["Predito 0", "Predito 1"], yticklabels=["Real 0", "Real 1"])
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Classes Preditas")
    plt.ylabel("Classes Reais")
    plt.show()

# Exibindo a matriz de confusão para cada modelo
for resultado in resultados:
    plot_matriz_confusao(resultado["Matriz de Confusão"], resultado["Modelo"])
