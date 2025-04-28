
# Harris Corner Detector com Avaliação de Robustez

Este repositório contém uma implementação **manual** do Detector de Cantos de Harris em Python, comparada com a versão pronta do OpenCV, e um protocolo de avaliação da robustez sob diferentes níveis de ruído gaussiano.

---

## 📚 Descrição

1. **Detector Manual**  
   - Cálculo de gradientes (Sobel)  
   - Montagem da matriz de segundo momento (M) com suavização gaussiana  
   - Cálculo da resposta \( R = \det(M) - k \,\mathrm{trace}(M)^2 \)  
   - Limiarização e supressão de não-máximos  

2. **Detector OpenCV**  
   - Usa `cv2.cornerHarris()`  
   - Mesmos parâmetros \(k\), blockSize e threshold para comparação

3. **Avaliação de Robustez**  
   - Adição de ruído gaussiano com \(\sigma \in \{0,5,10,20,30,40\}\)  
   - Métricas:
     - **RMS**: distância quadrática média entre cantos “verdadeiros” e detectados  
     - **Falsos Positivos** (Spurious)  
     - **Não Detectados** (Missed)

---

## ⚙️ Pré-requisitos

- Python 3.7+
- NumPy
- OpenCV (`pip install opencv-python`)
- SciPy
- Matplotlib
- (Opcional) Google Colab

---

## 🚀 Como Executar

### 1. Instalar dependências

```bash
pip install numpy opencv-python scipy matplotlib
