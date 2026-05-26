import base64
import io
from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def load_image_from_base64(b64_string):
    """Decodifica uma string Base64 e a carrega como uma imagem numpy."""
    try:
        # Corrigir padding se necessário
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += '=' * (4 - missing_padding)
        
        # Decodificar a string para bytes
        image_bytes = base64.b64decode(b64_string)
        
        # Criar um stream de bytes em memória
        image_buffer = io.BytesIO(image_bytes)
        
        # Abrir a imagem usando Pillow
        img = Image.open(image_buffer)
        
        # Converter para escala de cinza e para numpy array
        img_gray = img.convert('L')
        img_np = np.array(img_gray)
        
        # Normalizar a imagem para o intervalo [0, 1]
        img_normalized = img_np.astype(float) / 255.0
        
        return img_normalized
        
    except Exception as e:
        print(f"Erro ao carregar imagem base64: {e}")
        return None

# --- Criar dados borrados e ruidosos ---
def create_blur_operator(size):
    """Cria um kernel de 'borrão' (desfoque gaussiano)."""
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 0.5, 0.0
    kernel = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return kernel / np.sum(kernel)

# Ação do operador de borrão 'A' é uma convolução
def op_A(image, blur_kernel):
    return convolve2d(image, blur_kernel, mode='same', boundary='wrap')

# Ação do operador transposto 'A^T' (necessário para o gradiente)
def op_AT(image, blur_kernel):
    return convolve2d(image, np.rot90(blur_kernel, 2), mode='same', boundary='wrap')

# Parâmetros do problema
# LAMBDA_REG e NOISE_LEVEL podem precisar de ajuste para imagens reais

def obtem_imagem():

    # Leitura da imagem como String Base64 representando a imagem do problema
    with open('imagem.txt', 'r') as file:
        imagem_b64 = file.read().strip()

    NOISE_LEVEL = 0.05 # Nível de ruído adicionado à imagem borrada

    # Carregar a imagem a partir da string
    original_image = load_image_from_base64(imagem_b64)
    original_image = original_image[::64, ::64]
    IMAGE_SHAPE = original_image.shape
    IMAGE_SIZE_H, IMAGE_SIZE_W = IMAGE_SHAPE
    n_vars = IMAGE_SIZE_H * IMAGE_SIZE_W

    blur_kernel = create_blur_operator(9)

    # Criar a imagem borrada 'b'
    blurred_image = op_A(original_image, blur_kernel)

    # Adicionar ruído gaussiano
    np.random.seed(42)
    noise = np.random.normal(0, NOISE_LEVEL, blurred_image.shape)
    blurry_noisy_image = blurred_image + noise
    blurry_noisy_image = np.clip(blurry_noisy_image, 0, 1) # Garantir que os valores fiquem em [0,1]

    b_vec = blurry_noisy_image.flatten()

    return blurry_noisy_image, IMAGE_SHAPE, b_vec, blur_kernel