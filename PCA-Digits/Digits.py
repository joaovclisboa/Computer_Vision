import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Carregar o conjunto de dados
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)

# Pré-processamento: centralizar os dados
X_medio = np.mean(X_train, axis=0)
X_centrado = X_train - X_medio

# Aplicar PCA para redução de dimensionalidade
pca = PCA(0.9)
pca.fit(X_centrado)
autovetores = pca.components_

# Funções auxiliares
def calcular_coeficientes_medios(X_train, y_train, X_medio, autovetores, k):
    """Calcula coeficientes médios para cada classe (0-9)."""
    coef_medios = []
    for classe in range(10):
        X_classe = X_train[y_train == classe]
        coef_classe = [autovetores[:k] @ (amostra - X_medio) for amostra in X_classe]
        coef_medios.append(np.mean(coef_classe, axis=0))
    return np.array(coef_medios)

def classificar_digitos(X_test, X_medio, autovetores, coef_medios, k):
    """Classifica dígitos usando distância euclidiana."""
    y_pred = []
    for amostra in X_test:
        coef_amostra = autovetores[:k] @ (amostra - X_medio)
        distancias = [np.linalg.norm(coef_amostra - coef_classe) for coef_classe in coef_medios]
        y_pred.append(np.argmin(distancias))
    return np.array(y_pred)

def calcular_metricas(cm):
    """Calcula precisão e recall médios."""
    precisao_media = np.mean([cm[i, i] / cm[:, i].sum() for i in range(cm.shape[0]) if cm[:, i].sum() > 0])
    recall_medio = np.mean([cm[i, i] / cm[i, :].sum() for i in range(cm.shape[0]) if cm[i, :].sum() > 0])
    return precisao_media, recall_medio

# Executar classificação para diferentes valores de k
valores_k = [5, 10, 15, 20]
resultados = {}

for k in valores_k:
    coef_medios = calcular_coeficientes_medios(X_train, y_train, X_medio, autovetores, k)
    y_pred = classificar_digitos(X_test, X_medio, autovetores, coef_medios, k)
    cm = confusion_matrix(y_test, y_pred)
    precisao_media, recall_medio = calcular_metricas(cm)

    resultados[k] = {
        'matriz_confusao': cm,
        'Precisão Média': recall_medio,
        'Revocação Média': precisao_media,
        'Acurácia': np.trace(cm) / np.sum(cm),
        'y_pred': y_pred
    }

# Plotar matrizes de confusão
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, k in enumerate(valores_k):
    cm = resultados[k]['matriz_confusao']
    precisao = resultados[k]['Precisão Média']
    recall = resultados[k]['Revocação Média']
    acuracia = resultados[k]['Acurácia']

    im = axes[idx].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[idx].set_title(f'k={k}\nPrecisão: {precisao:.3f}\nRevocação: {recall:.3f}\nAcurácia: {acuracia:.3f}', fontsize=10)
    axes[idx].set_ylabel('Classe Verdadeira')
    axes[idx].set_xticks(np.arange(10))
    axes[idx].set_yticks(np.arange(10))
    axes[idx].set_xticklabels(range(10))
    axes[idx].set_yticklabels(range(10))

    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            axes[idx].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=8)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)

# Ajustar espaçamentos manualmente
plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)
plt.show()

# Resumo dos resultados
print("\n" + "="*50)
print("RESUMO DOS RESULTADOS")
print("="*50)
print(f"{'k':<5} {'Precisão Média':<20} {'Revocação Média':<20} {'Acurácia':<10}")
print("-" * 55)

for k in valores_k:
    precisao = resultados[k]['Precisão Média']
    recall = resultados[k]['Revocação Média']
    acuracia = resultados[k]['Acurácia']
    print(f"{k:<5} {precisao:<20.4f} {recall:<20.4f} {acuracia:<10.4f}")

# Gráficos comparativos das métricas
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(valores_k, [resultados[k]['Precisão Média'] for k in valores_k], 'bo-')
plt.title('Precisão Média vs K', fontsize=11)
plt.xlabel('Componentes (k)', fontsize=10)
plt.ylabel('Precisão Média', fontsize=10)
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(valores_k, [resultados[k]['Revocação Média'] for k in valores_k], 'ro-')
plt.title('Revocação Média vs K', fontsize=11)
plt.xlabel('Componentes (k)', fontsize=10)
plt.ylabel('Revocação Média', fontsize=10)
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(valores_k, [resultados[k]['Acurácia'] for k in valores_k], 'go-')
plt.title('Acurácia vs K', fontsize=11)
plt.xlabel('Componentes (k)', fontsize=10)
plt.ylabel('Acurácia', fontsize=10)
plt.grid(True)

plt.tight_layout(pad=2.0)
plt.show()
