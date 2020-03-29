import typing

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False


def open_img(path: str) -> np.ndarray:
    img = Image.open(path)
    return np.array(img)


def save_img(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)


def normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


def atmosph(img: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    m, n = img.shape
    k = 0.0025
    x = np.arange(-m // 2, m // 2, 1).reshape(m, 1)
    y = np.arange(-n // 2, n // 2, 1).reshape(1, n)
    t = (x ** 2 + y ** 2) ** (5 / 6)
    H = np.exp(-k * t)
    im_float = normalize(img.astype(np.float64))
    im_F = np.fft.fftshift(np.fft.fft2(im_float))
    im_blured_F = im_F * H
    im_blured = np.real(np.fft.ifft2(np.fft.ifftshift(im_blured_F)))
    im_blured = normalize(im_blured)
    im_blured = (im_blured * 255).round().astype(np.uint8)
    return H, im_blured


def motionblur(img: np.array, sigma: float) -> typing.Tuple[np.ndarray, np.ndarray]:
    m, n = img.shape
    a = 0.1
    b = 0.1
    T = 1
    x = np.arange(-m // 2, m // 2, 1).reshape(m, 1)
    y = np.arange(-n // 2, n // 2, 1).reshape(1, n)
    tmp1 = np.pi * (x * a + y * b)
    tmp2 = np.sin(tmp1) / tmp1
    tmp2 = np.nan_to_num(tmp2, nan=1, copy=False)  # 处理 /0 的情况
    H = T * tmp2 * np.exp(np.complex(0, -1) * tmp1)
    # 高斯噪声
    noise = np.random.normal(loc=0, scale=np.sqrt(sigma), size=(m, n))
    im_float = normalize(img.astype(np.float64))
    im_F = np.fft.fftshift(np.fft.fft2(im_float))
    noise_F = np.fft.fftshift(np.fft.fft2(noise))
    im_blured_F = im_F * H + noise_F
    im_blured = np.real(np.fft.ifft2(np.fft.ifftshift(im_blured_F)))
    im_blured = normalize(im_blured)
    im_blured = np.clip(im_blured, 0, 1)
    im_blured = (im_blured * 255).round().astype(np.uint8)
    return H, im_blured


def my_inverse(img: np.ndarray, H: np.ndarray, D0: float) -> typing.Tuple[np.ndarray, np.ndarray]:
    m, n = img.shape
    im_float = normalize(img.astype(np.float64))
    im_F = np.fft.fftshift(np.fft.fft2(im_float))
    im_reverse_F = im_F / H
    im_reverse = np.real(np.fft.ifft2(np.fft.ifftshift(im_reverse_F)))
    im_reverse = normalize(im_reverse)
    im_reverse = (im_reverse * 255).round().astype(np.uint8)
    x = np.arange(-m // 2, m // 2, 1).reshape(m, 1)
    y = np.arange(-n // 2, n // 2, 1).reshape(1, n)
    B = 1 / (1 + ((x ** 2 + y ** 2) / D0 ** 2) ** n)
    # butterworth 滤波
    im_reverse_b_F = im_reverse_F * B
    im_reverse_b = np.real(np.fft.ifft2(np.fft.ifftshift(im_reverse_b_F)))
    im_reverse_b = normalize(im_reverse_b)
    im_reverse_b = (im_reverse_b * 255).round().astype(np.uint8)
    return im_reverse, im_reverse_b


def my_wiener(img: np.ndarray, H: np.ndarray, K: float) -> np.ndarray:
    m, n = img.shape
    V = 1 / H * (np.abs(H) ** 2 / (np.abs(H) ** 2 + K))
    im_float = normalize(img.astype(np.float64))
    im_F = np.fft.fftshift(np.fft.fft2(im_float))
    im_wiener_F = im_F * V
    im_wiener = np.real(np.fft.ifft2(np.fft.ifftshift(im_wiener_F)))
    im_wiener = normalize(im_wiener)
    im_wiener = (im_wiener * 255).round().astype(np.uint8)
    return im_wiener


def exp5_28():
    img = open_img('img/demo-1.jpg')
    # 图像退化（大气湍流模型）
    # Output（H：退化模型， im_f：退化后图片）
    H, im_f = atmosph(img)
    # 全逆滤波，半径受限逆滤波
    D0 = 60
    # Input（im_f：退化图片，H：退化模型，D0：半径）
    # Output（im_inverse：全逆滤波结果，im_inverse_b：半径受限逆滤波）
    im_inverse, im_inverse_b = my_inverse(im_f, H, D0)
    # 维纳滤波
    K = 0.0001
    # Input（im_f：退化图片，H：退化模型，K：维纳滤波常数）
    im_wiener = my_wiener(im_f, H, K)
    save_img(im_inverse, 'result/exp5_28_inverse.jpg')
    save_img(im_inverse_b, 'result/exp5_28_inverse_b.jpg')
    save_img(im_wiener, 'result/exp5_28_wiener.jpg')
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('原图')
    axes[1].imshow(im_f, cmap='gray')
    axes[1].set_title('大气湍流(k=0.0025)')
    axes[2].imshow(im_inverse, cmap='gray')
    axes[2].set_title('全逆滤波')
    axes[3].imshow(im_inverse_b, cmap='gray')
    axes[3].set_title('半径受限的逆滤波')
    axes[4].imshow(im_wiener, cmap='gray')
    axes[4].set_title('维纳滤波')
    fig.tight_layout()
    fig.savefig('result/exp5_28.jpg')


def exp5_29():
    img = open_img('img/demo-2.jpg')
    # 图像退化（运动模糊+高斯噪声）
    _, im1_f = motionblur(img, 0.01)
    _, im2_f = motionblur(img, 0.001)
    H, im3_f = motionblur(img, 0.0000001)
    # 全逆滤波，半径受限逆滤波
    D0 = 33
    _, im1_inverse = my_inverse(im1_f, H, D0)
    _, im2_inverse = my_inverse(im2_f, H, D0)
    _, im3_inverse = my_inverse(im3_f, H, D0)
    # 维纳滤波
    K = 0.0001
    im1_wiener = my_wiener(im1_f, H, K)
    im2_wiener = my_wiener(im2_f, H, K)
    im3_wiener = my_wiener(im3_f, H, K)
    save_img(im1_f, 'result/exp5_29_im1_f.jpg')
    save_img(im2_f, 'result/exp5_29_im2_f.jpg')
    save_img(im3_f, 'result/exp5_29_im3_f.jpg')
    save_img(im1_inverse, 'result/exp5_29_im1_inverse.jpg')
    save_img(im2_inverse, 'result/exp5_29_im2_inverse.jpg')
    save_img(im3_inverse, 'result/exp5_29_im3_inverse.jpg')
    save_img(im1_wiener, 'result/exp5_29_im1_wiener.jpg')
    save_img(im2_wiener, 'result/exp5_29_im2_wiener.jpg')
    save_img(im3_wiener, 'result/exp5_29_im3_wiener.jpg')
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes[0, 0].imshow(im1_f, cmap='gray')
    axes[0, 0].set_title('运动模糊+加性噪声(sigma)')
    axes[0, 1].imshow(im1_inverse, cmap='gray')
    axes[0, 1].set_title('逆滤波结果')
    axes[0, 2].imshow(im1_wiener, cmap='gray')
    axes[0, 2].set_title('维纳滤波结果')
    axes[1, 0].imshow(im2_f, cmap='gray')
    axes[1, 0].set_title('运动模糊+加性噪声(sigma*0.1)')
    axes[1, 1].imshow(im2_inverse, cmap='gray')
    axes[1, 1].set_title('逆滤波结果')
    axes[1, 2].imshow(im2_wiener, cmap='gray')
    axes[1, 2].set_title('维纳滤波结果')
    axes[2, 0].imshow(im3_f, cmap='gray')
    axes[2, 0].set_title('运动模糊+加性噪声(sigma*0.00001)')
    axes[2, 1].imshow(im3_inverse, cmap='gray')
    axes[2, 1].set_title('逆滤波结果')
    axes[2, 2].imshow(im3_wiener, cmap='gray')
    axes[2, 2].set_title('维纳滤波结果')
    fig.tight_layout()
    fig.savefig('result/exp5_29.jpg')


if __name__ == '__main__':
    exp5_28()
    exp5_29()
