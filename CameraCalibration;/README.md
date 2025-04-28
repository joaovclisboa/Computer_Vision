# Calibração de Câmera: Toolbox vs. Implementação em Python

Este repositório apresenta um estudo comparativo entre duas abordagens para calibração de câmera:

- **Método 1: Octave Camera Calibration Toolbox**
- **Método 2: Implementação DLT + SVD em Python**

## Descrição do Projeto

A calibração de câmeras visa estimar os parâmetros que relacionam pontos 3D do mundo real aos pontos 2D de uma imagem:

1. **Parâmetros Intrínsecos**
   - Distância focal (f)
   - Tamanho do pixel (s<sub>x</sub>, s<sub>y</sub>)
   - Coeficientes de distorção radial (k<sub>1</sub>, k<sub>2</sub>)
   - Centro projetivo / ponto principal (c<sub>x</sub>, c<sub>y</sub>)

2. **Parâmetros Extrínsecos**
   - Matriz de rotação (R)
   - Vetor de translação (t)

Para evitar degenerações, utilizou-se uma configuração tridimensional de pontos em um tabuleiro de quadrados (26.5 mm × 26.5 mm), garantindo variações em X, Y e Z.

## Estrutura do Repositório

```
/                                      # raiz do projeto
├── data/                              # imagens e arquivos de pontos (3D e 2D)
│   ├── imagem_tabuleiro.jpg           # foto do tabuleiro tridimensional
│   └── pontos_3d_2d.csv               # correspondências world ↔ image
├── octave/                            # scripts e notebook Octave Toolbox
│   └── camera_calibration_octave.m    # processo passo a passo no Octave
├── python/                            # implementação própria em Python
│   ├── calibration_dlt_svd.py         # DLT, SVD, RQ e cálculo de C
│   └── utils.py                       # funções auxiliares (ruído, métricas)
├── notebooks/                         # Google Colab notebooks
│   ├── CameraCalibration_octave.ipynb # Notebook Octave Toolbox
│   └── CameraCalibration_python.ipynb # Notebook Python e análise de resultados
├── README.md                          # este arquivo
└── requirements.txt                   # dependências Python
```

## Método 1: Octave Camera Calibration Toolbox

1. Seleção das imagens de calibração
2. Definição dos eixos e pontos de referência no tabuleiro
3. Especificação do tamanho dos quadrados (26.5 mm)
4. Cálculo automático dos parâmetros intrínsecos e extrínsecos

*Saída típica (intrínsecos):*
```
K = [1115.59,    0,    799.60;
        0,   1116.31, 599.59;
        0,       0,      1   ]
```

## Método 2: DLT + SVD em Python

1. Conversão de coordenadas de quadrados (q_points) para milímetros, com ajustes de bordas (15 mm)
2. Construção da matriz A (2𝑛×12) pelo método DLT
3. Solução via SVD: última coluna de Vᵀ → vetor p (12×1)
4. Reshape para matriz de projeção P (3×4)
5. Decomposição RQ em `M = K·R` para obter K (intrínsecos) e R (rotação)
6. Cálculo do centro de projeção C = –M⁻¹·p<sub>4</sub>

## Como Executar

### Octave Toolbox

1. Abra o notebook `notebooks/CameraCalibration_octave.ipynb` no Octave.
2. Altere caminhos de imagem caso necessário.
3. Execute as células na ordem para obter K, R e t.

### Python (DLT + SVD)

1. Crie um ambiente virtual e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Coloque sua imagem do tabuleiro em `data/` e edite `utils.py` se mudar o nome.
3. Execute:
   ```bash
   python python/calibration_dlt_svd.py --input data/imagem_tabuleiro.jpg
   ```
4. Os parâmetros K, R e o centro C serão impressos no console.

## Dependências

- Python 3.7+
- NumPy
- SciPy
- OpenCV (`opencv-python`)
- Matplotlib (para plots nos notebooks)

## Autor

Desenvolvido por **Seu Nome** – Abril 2025

---

*Este projeto foi desenvolvido como parte de estudo de calibração de câmeras e compara métodos analíticos (Octave Toolbox) com implementação numérica (Python, DLT+SVD).*


