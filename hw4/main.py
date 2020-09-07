from typing import List, Tuple

import cv2
import numpy as np


def normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


def draw_histogram(img: np.ndarray, ax):
    level = int(2 ** (8 * img.nbytes / img.size))
    ax.hist(img.flatten(), bins=level, density=True, histtype='stepfilled')


# 双线性插值
def interpolation_bilinear(img: np.ndarray, scale: float) -> np.ndarray:
    dst_shape = (round(img.shape[0] * scale), round(img.shape[1] * scale)) + img.shape[2:]
    dst_x = np.arange(0, dst_shape[0])
    dst_y = np.arange(0, dst_shape[1])
    src_x = (dst_x + 0.5) / scale - 0.5
    src_y = (dst_y + 0.5) / scale - 0.5
    # 四个最近邻点，缩小时会出现 src 坐标超出范围，用 max min 约束
    src_x_0 = np.maximum(np.floor(src_x), 0).astype(int)
    src_x_1 = np.minimum(np.ceil(src_x), img.shape[0] - 1).astype(int)
    src_y_0 = np.maximum(np.floor(src_y), 0).astype(int)
    src_y_1 = np.minimum(np.ceil(src_y), img.shape[1] - 1).astype(int)
    pixel_00 = img[np.ix_(src_x_0, src_y_0)]
    pixel_01 = img[np.ix_(src_x_0, src_y_1)]
    pixel_10 = img[np.ix_(src_x_1, src_y_0)]
    pixel_11 = img[np.ix_(src_x_1, src_y_1)]
    delta_x = src_x - src_x_0
    delta_y = src_y - src_y_0
    delta_x, delta_y = np.ix_(delta_x, delta_y)
    # 同时支持单通道和多通道图片
    s = dst_shape[:2] + (1,) * (len(dst_shape) - 2)
    dst = ((1 - delta_x) * (1 - delta_y)).reshape(s) * pixel_00 \
          + ((1 - delta_x) * delta_y).reshape(s) * pixel_01 \
          + (delta_x * (1 - delta_y)).reshape(s) * pixel_10 \
          + (delta_x * delta_y).reshape(s) * pixel_11
    return dst.astype(img.dtype)


# 高斯低通滤波
def gauss_low_pass_filtering(img: np.ndarray, D0: float) -> np.ndarray:
    m, n = img.shape[:2]
    x = np.arange(-m // 2, m // 2, 1).reshape(m, 1)
    y = np.arange(-n // 2, n // 2, 1).reshape(1, n)
    H = np.exp(-(x ** 2 + y ** 2) / (2 * D0 ** 2))
    H = H.reshape((m, n) + (1,) * (img.ndim - 2))
    im_float = img.astype(np.float64)
    im_F = np.fft.fftshift(np.fft.fft2(im_float, axes=(0, 1)), axes=(0, 1))
    im_flitered_F = im_F * H
    im_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(im_flitered_F, axes=(0, 1)), axes=(0, 1)))
    im_filtered = np.clip(im_filtered, a_min=0, a_max=255)
    im_filtered = im_filtered.round()
    return im_filtered.astype(np.uint8)


# 残差金字塔
# 近似滤波器使用低通高斯平滑滤波器
# 插值滤波器使用双线性插值
def residual_pyramid(img: np.ndarray, level: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    pyramid = []
    residual = []
    img_now = img
    for _ in range(level):
        pyramid.append(img_now)
        img_filtered = gauss_low_pass_filtering(img_now, D0=img_now.shape[0] / 2)
        img_level_up = img_filtered[::2, ::2]
        img_level_up_down = interpolation_bilinear(img_level_up, 2)
        res = (img_level_up_down - img_now) + 127
        residual.append(res)
        img_now = img_level_up
    return pyramid, residual


def fast_wavelet_transform(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_phi = (0.0322, -0.0126, -0.0992, 0.2979, 0.8037, 0.4976, -0.0296, -0.0758)


def exp1():
    img = cv2.imread('img/demo-1.jpg')
    pyramid, residual = residual_pyramid(img, 4)
    for i in range(4):
        cv2.imwrite(f'result/pyramid_{i}.png', pyramid[i])
        cv2.imwrite(f'result/residual_{i}.png', residual[i])


def exp2():
    img = cv2.imread('img/demo-2.tif')


if __name__ == '__main__':
    exp1()
    exp2()
