import numpy as np
import cv2
import matplotlib.pyplot as plt

# Tamanho de cada quadrado
square_size = 50

# Número de quadrados em cada dimensão
num_rows = 8
num_cols = 8

# Criar a imagem vazia
img_size = (num_rows * square_size, num_cols * square_size)
chessboard = np.zeros(img_size, dtype=np.uint8)

# Preencher os quadrados
for row in range(num_rows):
    for col in range(num_cols):
        if (row + col) % 2 == 0:
            y_start = row * square_size
            y_end   = y_start + square_size
            x_start = col * square_size
            x_end   = x_start + square_size
            chessboard[y_start:y_end, x_start:x_end] = 255  # Branco

# Mostrar a imagem
plt.imshow(chessboard, cmap='gray')
plt.axis('off')
cv2.imwrite("image.jpg", chessboard)
img_gray = np.float32(img)
plt.show()
from scipy.ndimage import maximum_filter

# --- Parâmetros de Experimento ---

results = {}     # Resultados do detector manual
results_cv = {}  # Resultados do detector do OpenCV

# Conjuntos de parâmetros para teste
ks = [0.04, 0.02]                 # Parâmetros k do detector de Harris
thresholds = [0.01, 0.02]         # Limiares relativos para filtrar respostas
sigmas = [0, 5, 10, 20, 30, 40]   # Níveis de ruído gaussiano

blockSize = 5   # Tamanho da vizinhança para o cálculo da matriz M
ksize_ = 3      # Tamanho do kernel do Sobel (gradientes)

# --- Função do Detector Manual de Harris ---
def harris_detector(img, k=0.4, window_size=blockSize, threshold_rel=0.01):
    """
    Implementação manual do detector de Harris.
    """
    I = img.astype(np.float32)
    
    # Gradientes em x e y (usando Sobel)
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=ksize_)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=ksize_)
    
    # Elementos da matriz de Harris (produto dos gradientes)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Aplicação de suavização gaussiana
    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)

    # Cálculo da resposta R de Harris
    detM = Sxx * Syy - Sxy * Sxy
    traceM = Sxx + Syy
    R = detM - k * (traceM**2)

    # Limiarização e supressão de não-máximos
    thresh = threshold_rel * R.max()
    R_thr = (R > thresh) * R
    local_max = maximum_filter(R_thr, size=3)

    # Obtenção das coordenadas dos cantos
    coords = np.argwhere((R_thr == local_max) & (R_thr != 0))
    corners = [tuple(pt[::-1]) for pt in coords]  # Inverte (y,x) -> (x,y)
    
    return corners


# --- Função com OpenCV ---

def harris_opencv_corners(img, blockSize, ksize, k, threshold_rel, nms_size=3):
    """
    Detector de Harris usando OpenCV (cv2.cornerHarris).
    """
    gray = np.float32(img)
    R = cv2.cornerHarris(gray, blockSize, ksize, k)

    # Limiarização e supressão de não-máximos
    thresh = threshold_rel * R.max()
    R_thr = (R > thresh) * R
    local_max = maximum_filter(R_thr, size=nms_size)

    pts = np.argwhere((R_thr == local_max) & (R_thr != 0))
    return [tuple(p[::-1]) for p in pts]


# --- Ruído Gaussiano ---

def add_gaussian_noise(img, sigma):
    """
    Adiciona ruído gaussiano com desvio padrão sigma.
    """
    noise = np.random.normal(0, sigma, img.shape)
    noisy_img = np.clip(img + noise, 0, 255)
    return noisy_img.astype(np.uint8)


# --- Métricas ---

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_rms(detected_corners, true_corners):
    """
    Calcula o erro RMS entre os cantos detectados e os verdadeiros.
    """
    distances = []
    for true_corner in true_corners:
        closest_distance = float('inf')
        for detected_corner in detected_corners:
            distance = euclidean_distance(true_corner, detected_corner)
            if distance < closest_distance:
                closest_distance = distance
        distances.append(closest_distance)

    rms = np.sqrt(np.mean(np.array(distances)**2)) if distances else float('inf')
    return rms


def count_spurious_and_missed(true_corners, detected_corners, threshold=5):
    """
    Conta cantos espúrios (falsos positivos) e não detectados (falsos negativos).
    """
    spurious = 0
    missed = 0

    # Cantos detectados que não estão próximos de nenhum verdadeiro
    for detected_corner in detected_corners:
        is_spurious = True
        for true_corner in true_corners:
            if euclidean_distance(detected_corner, true_corner) < threshold:
                is_spurious = False
                break
        if is_spurious:
            spurious += 1

    # Cantos verdadeiros que não foram detectados
    for true_corner in true_corners:
        is_missed = True
        for detected_corner in detected_corners:
            if euclidean_distance(true_corner, detected_corner) < threshold:
                is_missed = False
                break
        if is_missed:
            missed += 1

    return spurious, missed


# --- Loop Principal de Avaliação ---

# Itera sobre combinações de k e threshold
for k in ks:
    for thr in thresholds:
        key = f"k{k}_thr{thr}"  # Nome da configuração

        # Inicializa dicionários para armazenar os resultados
        results[key] = {
            'sigma':     list(sigmas),
            'rms':       [None]*len(sigmas),
            'spurious':  [None]*len(sigmas),
            'missed':    [None]*len(sigmas),
        }
        results_cv[key] = {
            'sigma':     list(sigmas),
            'rms':       [None]*len(sigmas),
            'spurious':  [None]*len(sigmas),
            'missed':    [None]*len(sigmas),
        }

        # Cantos da imagem original (sem ruído)
        true_corners = harris_detector(img_gray, k=k, threshold_rel=thr)
        true_cv = harris_opencv_corners(img_gray, blockSize, ksize_, k, thr)

        # Para cada nível de ruído
        for idx, sigma in enumerate(sigmas):
            noisy = add_gaussian_noise(img_gray, sigma)

            # Detecta cantos nas imagens com ruído
            detected = harris_detector(noisy, k=k, threshold_rel=thr)
            det_cv = harris_opencv_corners(noisy, blockSize, ksize_, k, thr)

            # Calcula métricas
            rms = calculate_rms(true_corners, detected)
            sp, ms = count_spurious_and_missed(true_corners, detected)

            rms_cv = calculate_rms(true_cv, det_cv)
            sp_cv, ms_cv = count_spurious_and_missed(true_cv, det_cv)

            # Armazena os resultados na posição correspondente
            results[key]['rms'][idx] = rms
            results[key]['spurious'][idx] = sp
            results[key]['missed'][idx] = ms

            results_cv[key]['rms'][idx] = rms_cv
            results_cv[key]['spurious'][idx] = sp_cv
            results_cv[key]['missed'][idx] = ms_cv

# --- Visualização dos Gráficos ---
metrics = ['rms', 'spurious', 'missed']
titles  = {'rms':'RMS','spurious':'Falsos Positivos','missed':'Não Detectados'}

for metric in metrics:
    # 1 linha, 2 colunas; compartilha o eixo Y
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4), sharey=True)

    # colo 1:
    ax1.set_title("Minha Implementação")
    for key, data in results.items():
        ax1.plot(data['sigma'], data[metric], 'o-', label=key)
    ax1.set_xlabel('σ do ruído')
    ax1.set_ylabel(titles[metric])
    ax1.legend(fontsize='small', ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # colo 2: OpenCV
    ax2.set_title("OpenCV (cornerHarris)")
    for key, data in results_cv.items():
        ax2.plot(data['sigma'], data[metric], 'x--', label=key)
    ax2.set_xlabel('σ do ruído')
    # só legenda, sem y-label duplicado
    ax2.legend(fontsize='small', ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # título geral e ajuste de layout
    fig.suptitle(f"{titles[metric]} vs σ do Ruído", y=1.03, fontsize=14)
    plt.tight_layout()
    plt.show()
