import numpy as np
import matplotlib.pyplot as plt
from utils import obtem_imagem, op_A

# Pequeno valor para estabilidade numérica no gradiente da TV
LAMBDA_REG = 2e-2  # Parâmetro de regularização lambda
EPSILON = 1e-8

# --- 2. Carregamento da Imagem ---
imagem_inicial, IMAGE_SHAPE, b_vec, blur_kernel = obtem_imagem()

# --- 3. Funções de Otimização ---
def total_variation(x_vec):
    x_2d = x_vec.reshape(IMAGE_SHAPE)
    grad_x = np.roll(x_2d, -1, axis=1) - x_2d
    grad_y = np.roll(x_2d, -1, axis=0) - x_2d
    return np.sum(np.sqrt(grad_x**2 + grad_y**2 + EPSILON))

def objective_function(x_vec):
    x_2d = x_vec.reshape(IMAGE_SHAPE)
    fidelity_term = np.sum((op_A(x_2d, blur_kernel).flatten() - b_vec)**2)
    regularization_term = LAMBDA_REG * total_variation(x_vec)
    return fidelity_term + regularization_term

# --- 4. Execução da Otimização ---
x0 = imagem_inicial.flatten()

solucao = # Essa variável é preenchida com o resultado da otimização usando o algoritmo escolhido

reconstructed_image = solucao.x.reshape(IMAGE_SHAPE)

# --- 5. Visualização ---

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].imshow(imagem_inicial, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Imagem Borrada e Ruidosa (Entrada)')
axes[0].axis('off')

axes[1].imshow(reconstructed_image, cmap='gray', vmin=0, vmax=1)
axes[1].set_title(f'Imagem Reconstruída')
axes[1].axis('off')

plt.tight_layout()
plt.show()