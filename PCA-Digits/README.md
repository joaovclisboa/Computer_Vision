# Classificação de Dígitos com PCA e Distância Euclidiana

Este projeto realiza a **classificação de dígitos manuscritos** do dataset `load_digits` da biblioteca `scikit-learn` usando **redução de dimensionalidade via PCA (Análise de Componentes Principais)** e **classificação baseada em distância euclidiana** no espaço dos autovetores.

## 📌 Objetivo

Explorar o uso de PCA para extrair características relevantes dos dados e avaliar o desempenho da classificação para diferentes números de componentes principais (`k`).

---

## 🧠 Tecnologias e Bibliotecas Utilizadas

- Python 3
- NumPy
- Matplotlib
- scikit-learn

---

## 📂 Estrutura do Código

1. **Carregamento e pré-processamento dos dados**
   - Dataset: `load_digits()`
   - Divisão em treino e teste (90%/10%)
   - Centralização dos dados (subtração da média)

2. **Redução de dimensionalidade com PCA**
   - `PCA(0.9)` retém 90% da variância total
   - Armazenamento dos autovetores (componentes principais)

3. **Classificação**
   - Para cada valor de `k` (número de componentes), são:
     - Calculados os coeficientes médios por classe
     - Calculadas as previsões no conjunto de teste
     - Geradas as métricas: **Precisão**, **Revocação** e **Acurácia**
     - Exibidas as **matrizes de confusão**

4. **Visualizações**
   - Matrizes de confusão para cada `k`
   - Gráficos comparativos das métricas (Precisão, Revocação, Acurácia)

---

## 📊 Resultados

O código exibe:

- Matrizes de confusão para `k = 5, 10, 15, 20`
- Métricas médias para cada valor de `k`
- Gráficos comparativos entre as métricas

---

## 🚀 Como Executar

1. Clone este repositório ou copie o código para seu ambiente local.
2. Instale as dependências:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```
3. Execute o script:
   ```bash
   python nome_do_arquivo.py
   ```

---

## 📌 Exemplos de Métricas Impressas

```
==================================================
RESUMO DOS RESULTADOS
==================================================
k     Precisão Média      Revocação Média     Acurácia  
-------------------------------------------------------
5     0.9312              0.9267              0.9333    
10    0.9441              0.9398              0.9500    
15    0.9612              0.9575              0.9611    
20    0.9665              0.9633              0.9667    
```

---

## 📎 Notas

- O classificador é simples e **não supervisionado**, baseado apenas na **distância euclidiana** entre os coeficientes projetados no espaço de menor dimensão.
- O uso de `PCA(0.9)` garante que os autovetores iniciais explicam pelo menos 90% da variância dos dados.
