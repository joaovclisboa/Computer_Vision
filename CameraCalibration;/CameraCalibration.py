import numpy as np
import scipy.linalg  # Para decomposição RQ usada na separação de K e R

# --- Pontos no tabuleiro (em unidades de quadrados) ---
# Cada linha: [coluna, linha, camada] no sistema de coordenadas do tabuleiro
q_points = np.array([
    [1, 1, 0],
    [2, 4, 0],
    [5, 4, 0],
    [6, 1, 0],
    [8, 5, 0],
    [2, 7, 1],
    [3, 7, 3],
    [6, 7, 1],
    [4, 7, 5],
    [6, 7, 4],
    [8, 7, 6],
    [9, 5, 6],
    [9, 2, 5],
    [9, 3, 3],
    [9, 5, 2],
    [9, 0, 2]
])

# --- Fator de escala (tamanho de cada quadrado em mm) ---
scale = 26.5

# Converte q_points para coordenadas do mundo real (mm)
world_points = []
for point in q_points:
    x, y, z = point * scale  # escala linear nos três eixos

    # Ajustes adicionais conforme especificação:
    # - Se y ultrapassa 6 quadrados (linha >6), adiciona 15 mm de borda
    if y > (6 * scale):
        y += 15

    # - Se estivermos acima do plano (z>0), adiciona 15 mm na vertical
    if z > 0:
        z += 15

    # - Se x ultrapassa 8 quadrados (coluna >8), adiciona 15 mm de borda
    if x > (8 * scale):
        x += 15

    world_points.append([x, y, z])

world_points = np.array(world_points)  # transforma em matriz NumPy (16×3)

# --- Pontos de imagem correspondentes (pixels) ---
i_points = np.array([
    [675, 831],
    [651, 741],
    [796, 710],
    [929, 767],
    [887, 674],
    [578, 611],
    [626, 513],
    [750, 589],
    [674, 411],
    [758, 454],
    [844, 356],
    [980, 381],
    [1081, 457],
    [1026, 546],
    [957, 570],
    [1134, 644]
])

num_points = len(world_points)  # número de correspondências (16)

# --- Montagem da matriz A para DLT (Direct Linear Transform) ---
# A terá 2 linhas por ponto, 12 colunas
A = np.zeros((2 * num_points, 12))

for idx in range(num_points):
    X, Y, Z = world_points[idx]
    u, v = i_points[idx]

    # Cada ponto gera duas linhas na matriz A:
    # [ -X -Y -Z -1   0  0  0  0   uX  uY  uZ  u ]
    # [  0  0  0  0  -X -Y -Z -1   vX  vY  vZ  v ]
    A[2 * idx]     = [-X, -Y, -Z, -1,  0,  0,  0,  0, u*X, u*Y, u*Z, u]
    A[2 * idx + 1] = [ 0,  0,  0,  0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v]

# --- Estimação da matriz de projeção P via SVD ---
# SVD de A: A = U·S·Vᵀ → último vetor singular de V (linha de Vt) minimiza ||A·p||
U, S, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)  # vetor p de 12 elementos → matriz 3×4

print("Matriz de projeção P (3×4):")
print(P)

# --- Extração da submatriz M (3×3 dos intrínsecos × rotação) ---
M = P[:, :3]   # os primeiros 3 colunas de P

# --- Decomposição RQ para obter K (intrínsecos) e R (rotação) ---
K, R = scipy.linalg.rq(M)

# Ajusta sinais para que K tenha valores positivos na diagonal
T = np.diag(np.sign(np.diag(K)))
K = K @ T
R = T @ R

# Normaliza K para que K[2,2] = 1
K /= K[2, 2]

print("\nMatriz intrínseca K:")
print(K)

print("\nMatriz de rotação R:")
print(R)

# --- Cálculo do centro da câmera C ---
# P = [M | p4], então C = -M⁻¹·p4
p4 = P[:, 3]                # última coluna de P
C = -np.linalg.inv(M) @ p4  # centro em coordenadas do mundo

print("\nCentro da câmera C (X, Y, Z):")
print(C)
