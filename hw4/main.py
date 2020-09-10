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


def fwt_convolution_row(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert img.ndim == 2
    assert kernel.ndim == 1
    assert kernel.size % 2 == 0
    img_pad = np.pad(img, pad_width=[(0, 0), (kernel.size - 1, kernel.size - 1)], mode='constant', constant_values=0)
    new_shape = (img.shape[0], img.shape[1] + kernel.size - 1, kernel.size)
    new_stride = (img_pad.strides[0], img_pad.strides[1], img_pad.strides[1])
    tmp = np.lib.stride_tricks.as_strided(img_pad, shape=new_shape, strides=new_stride)
    result = (tmp * kernel.reshape((1, 1, kernel.size))).sum(axis=2)
    # 下采样
    return result[:, kernel.size // 2:-(kernel.size // 2):2].copy()


def fwt_convolution_col(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert img.ndim == 2
    assert kernel.ndim == 1
    assert kernel.size % 2 == 0
    img_pad = np.pad(img, pad_width=[(kernel.size - 1, kernel.size - 1), (0, 0)], mode='constant', constant_values=0)
    new_shape = (img.shape[0] + kernel.size - 1, img.shape[1], kernel.size)
    new_stride = (img_pad.strides[0], img_pad.strides[1], img_pad.strides[0])
    tmp = np.lib.stride_tricks.as_strided(img_pad, shape=new_shape, strides=new_stride)
    result = (tmp * kernel.reshape((1, 1, kernel.size))).sum(axis=2)
    # 下采样
    return result[kernel.size // 2:-(kernel.size // 2):2, :].copy()


def ifwt_convolution_row(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert img.ndim == 2
    assert kernel.ndim == 1
    assert kernel.size % 2 == 0
    img_up = np.zeros(shape=(img.shape[0], 2 * img.shape[1]), dtype=img.dtype)
    img_up[:, ::2] = img
    img = img_up
    img_pad = np.pad(img, pad_width=[(0, 0), (kernel.size - 1, kernel.size - 1)], mode='constant', constant_values=0)
    new_shape = (img.shape[0], img.shape[1] + kernel.size - 1, kernel.size)
    new_stride = (img_pad.strides[0], img_pad.strides[1], img_pad.strides[1])
    tmp = np.lib.stride_tricks.as_strided(img_pad, shape=new_shape, strides=new_stride)
    result = (tmp * kernel.reshape((1, 1, kernel.size))).sum(axis=2)
    return result[:, kernel.size // 2:-(kernel.size // 2) + 1].copy()


def ifwt_convolution_col(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert img.ndim == 2
    assert kernel.ndim == 1
    assert kernel.size % 2 == 0
    img_up = np.zeros(shape=(2 * img.shape[0], img.shape[1]), dtype=img.dtype)
    img_up[::2, :] = img
    img = img_up
    img_pad = np.pad(img, pad_width=[(kernel.size - 1, kernel.size - 1), (0, 0)], mode='constant', constant_values=0)
    new_shape = (img.shape[0] + kernel.size - 1, img.shape[1], kernel.size)
    new_stride = (img_pad.strides[0], img_pad.strides[1], img_pad.strides[0])
    tmp = np.lib.stride_tricks.as_strided(img_pad, shape=new_shape, strides=new_stride)
    result = (tmp * kernel.reshape((1, 1, kernel.size))).sum(axis=2)
    return result[kernel.size // 2:-(kernel.size // 2) + 1, :].copy()


def fwt(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_phi = np.array([0.0322, -0.0126, -0.0992, 0.2979, 0.8037, 0.4976, -0.0296, -0.0758])
    g_phi = h_phi[::-1]
    g_psi = (-1) ** np.arange(0, g_phi.size) * g_phi[::-1]
    h_psi = g_psi[::-1]
    w_psi = fwt_convolution_col(img, h_psi[::-1])
    w_phi = fwt_convolution_col(img, h_phi[::-1])
    W = fwt_convolution_row(w_phi, h_phi[::-1])
    W_H = fwt_convolution_row(w_phi, h_psi[::-1])
    W_V = fwt_convolution_row(w_psi, h_phi[::-1])
    W_D = fwt_convolution_row(w_psi, h_psi[::-1])
    return W, W_H, W_V, W_D


def ifwt(W: np.ndarray, W_H: np.ndarray, W_V: np.ndarray, W_D: np.ndarray) -> np.ndarray:
    h_phi = np.array([0.0322, -0.0126, -0.0992, 0.2979, 0.8037, 0.4976, -0.0296, -0.0758])
    g_phi = h_phi[::-1]
    g_psi = (-1) ** np.arange(0, g_phi.size) * g_phi[::-1]
    h_psi = g_psi[::-1]
    w_psi = ifwt_convolution_row(W_D, h_psi[::-1]) + ifwt_convolution_row(W_V, h_phi[::-1])
    w_phi = ifwt_convolution_row(W_H, h_psi[::-1]) + ifwt_convolution_row(W, h_phi[::-1])
    return ifwt_convolution_col(w_psi, h_psi[::-1]) + ifwt_convolution_col(w_phi, h_phi[::-1])


def post_process(img: np.ndarray) -> np.ndarray:
    return ((img - img.min()) / (img.max() - img.min()) * 255).round().astype(np.uint8)


def combine_imgs(W: np.ndarray, W_H: np.ndarray, W_V: np.ndarray, W_D: np.ndarray) -> np.ndarray:
    assert W.shape == W_H.shape
    assert W.shape == W_V.shape
    assert W.shape == W_D.shape
    out = np.empty(shape=(W.shape[0] * 2, W.shape[1] * 2))
    out[:W.shape[0], :W.shape[1]] = W
    out[:W.shape[0], W.shape[1]:] = W_H
    out[W.shape[0]:, :W.shape[1]] = W_V
    out[W.shape[0]:, W.shape[1]:] = W_D
    return out


def exp1():
    img = cv2.imread('img/demo-1.jpg', flags=cv2.IMREAD_GRAYSCALE)
    pyramid, residual = residual_pyramid(img, 4)
    for i in range(4):
        cv2.imwrite(f'result/pyramid_{i}.png', pyramid[i])
        cv2.imwrite(f'result/residual_{i}.png', residual[i])


def exp2():
    img = cv2.imread('img/demo-2.tif', flags=cv2.IMREAD_GRAYSCALE)
    W_pre = img
    out_all = np.empty_like(W_pre)
    out = out_all
    Ws = []
    W_Hs = []
    W_Vs = []
    W_Ds = []
    for i in range(3):
        W, W_H, W_V, W_D = fwt(W_pre)
        Ws.append(W)
        W_Hs.append(W_H)
        W_Vs.append(W_V)
        W_Ds.append(W_D)
        out[:, :] = combine_imgs(post_process(W), post_process(W_H), post_process(W_V), post_process(W_D))
        cv2.imwrite(f'result/demo-2-fwt-{i + 1}.png', out_all)
        out = out[:W.shape[0], :W.shape[1]]
        W_pre = W
    # 边缘检测
    cv2.imwrite('result/demo2-edge-detect-1.png',
                combine_imgs(
                    combine_imgs(np.zeros_like(Ws[1]), post_process(W_Hs[1]),
                                 post_process(W_Vs[1]), post_process(W_Ds[1])),
                    post_process(W_Hs[0]), post_process(W_Vs[0]), post_process(W_Ds[0])
                )
                )
    W0 = ifwt(np.zeros_like(Ws[1]), W_Hs[1], W_Vs[1], W_Ds[1])
    out = ifwt(W0, W_Hs[0], W_Vs[0], W_Ds[0])
    cv2.imwrite('result/demo2-edge-detect-1-out.png', post_process(out))
    cv2.imwrite('result/demo2-edge-detect-2.png',
                combine_imgs(
                    combine_imgs(np.zeros_like(Ws[1]), np.zeros_like(W_Hs[1]),
                                 post_process(W_Vs[1]), post_process(W_Ds[1])),
                    np.zeros_like(W_Hs[0]), post_process(W_Vs[0]), post_process(W_Ds[0])
                )
                )
    W0 = ifwt(np.zeros_like(Ws[1]), np.zeros_like(W_Hs[1]), W_Vs[1], W_Ds[1])
    out = ifwt(W0, np.zeros_like(W_Hs[0]), W_Vs[0], W_Ds[0])
    cv2.imwrite('result/demo2-edge-detect-2-out.png', post_process(out))


if __name__ == '__main__':
    exp1()
    exp2()
