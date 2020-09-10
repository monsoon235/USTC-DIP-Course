import math
from typing import Iterable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 二值化
def binarize(img: np.ndarray) -> np.ndarray:
    level = int(2 ** (8 * img.nbytes / img.size))
    out = np.zeros_like(img, dtype=np.bool)
    out[img >= level / 2] = True
    return out


# 重建图像
def debinarize(img_bin: np.ndarray, level: int) -> np.ndarray:
    img = np.zeros_like(img_bin, dtype=np.uint8)
    img[img_bin] = level - 1
    return img


# 画直方图
def draw_histogram(img: np.ndarray, ax):
    level = int(2 ** (8 * img.nbytes / img.size))
    ax.hist(img.flatten(), bins=level, range=[0, level - 1], density=True, histtype='stepfilled')
    plt.xlabel('gray scale value')
    plt.ylabel('frequency')
    ax.xlim(0, level - 1)
    ax.ylim(bottom=0)


# 使用 CNN 的 im2col 技巧并行化计算
def im2col(im: np.ndarray, kernel_size, stride, inner_stride=(1, 1)) -> np.ndarray:
    kh, kw = kernel_size
    sh, sw = stride
    ish, isw = inner_stride
    h, w = im.shape[0:2]
    assert (h - kh * ish) % sh == 0
    assert (w - kw * isw) % sw == 0
    out_h = (h - kh * ish) // sh + 1
    out_w = (w - kw * isw) // sw + 1
    out_shape = (out_h, out_w, kh, kw) + im.shape[2:]
    s = im.strides
    out_stride = (s[0] * sh, s[1] * sw, s[0] * ish, s[1] * isw) + s[2:]
    col_img = np.lib.stride_tricks.as_strided(im, shape=out_shape, strides=out_stride)
    return col_img


# 高斯模糊
def gauss_blur(img: np.ndarray, kernel_size, sigma) -> np.ndarray:
    ksx, ksy = kernel_size
    x = np.arange(-ksx // 2, ksx // 2).reshape(ksx, 1)
    y = np.arange(-ksy // 2, ksy // 2).reshape(1, ksy)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    img_pad = np.pad(img, pad_width=[(ksx // 2, ksx // 2), (ksy // 2, ksy // 2)], mode='edge')
    img_pad_im2col = im2col(img_pad, kernel_size=kernel_size, stride=(1, 1))
    result = np.tensordot(img_pad_im2col, kernel, axes=[(2, 3), (0, 1)])
    return result


# Canny 算法
def canny(img: np.ndarray, kernel_size: Iterable[int], sigma: float, TL: float, TH: float) -> np.ndarray:
    assert img.ndim == 2
    level = int(2 ** (8 * img.nbytes / img.size))
    # 用一个高斯滤波器平滑输入图像
    img_smooth = gauss_blur(img, kernel_size, sigma)
    # 计算梯度幅值图像和角度图像
    kernel_x = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    kernel_y = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    img_smooth_pre_im2col = np.pad(img_smooth, pad_width=[(1, 1), (1, 1)], mode='edge')
    img_smooth_im2col = im2col(img_smooth_pre_im2col, (3, 3), (1, 1))
    gx = np.tensordot(img_smooth_im2col, kernel_x, axes=[(2, 3), (0, 1)])
    gy = np.tensordot(img_smooth_im2col, kernel_y, axes=[(2, 3), (0, 1)])
    M = np.sqrt(gx ** 2 + gy ** 2)
    alpha = np.arctan2(gy, gx)
    # 对梯度幅值图像应用非最大抑制
    selected_axis = (alpha / (np.pi / 4)).round() % 4
    neighbor = np.empty(shape=M.shape + (2,), dtype=M.dtype)
    M_pad = np.pad(M, pad_width=[(1, 1), (1, 1)], mode='constant', constant_values=0)
    x_delta = [[1, 1, 0, -1], [-1, -1, 0, 1]]
    y_delta = [[0, 1, 1, 1], [0, -1, -1, -1]]
    for i in range(4):
        xs, ys = np.where(selected_axis == i)
        neighbor[xs, ys, 0] = M_pad[xs + 1 + x_delta[0][i], ys + 1 + y_delta[0][i]]
        neighbor[xs, ys, 1] = M_pad[xs + 1 + x_delta[1][i], ys + 1 + y_delta[1][i]]
    gN = M.copy()
    gN[(M < neighbor[:, :, 0]) | (M < neighbor[:, :, 1])] = 0
    # 用双阈值处理和连接分析来检测并连接边缘
    gNH = gN >= (TH * (level - 1))
    gNL = gN >= (TL * (level - 1))
    gNH_pad = np.pad(gNH, pad_width=[(1, 1), (1, 1)])
    pxs, pys = np.where(gNH)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if not (i == 0 and j == 0):
                gNH_pad[pxs + 1 + i, pys + 1 + j] = True
    result = gNH_pad[1:-1, 1:-1] & gNL
    return result


def hough_transform(img: np.ndarray, kernel_size: Iterable[int], sigma: float, TL: float, TH: float) -> np.ndarray:
    assert img.ndim == 2
    level = int(2 ** (8 * img.nbytes / img.size))
    # 得到一幅二值图像
    img_canny_bin = canny(img, kernel_size, sigma, TL, TH)
    # 指定ρ-θ平面中的细分
    param_space = np.zeros(shape=(90 * 2, math.ceil(math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))), dtype=np.int)
    # 对像素高度集中的地方检验其累加单元的数量
    theta_deg = np.arange(-90, 90)
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    for x, y in zip(*np.where(img_canny_bin)):
        ro = (x * cos_theta + y * sin_theta).round().astype(np.int)
        param_space[theta_deg + 90, ro] += 1
    return param_space


def threshold_process_global(img: np.ndarray, threshold: int) -> np.ndarray:
    level = int(2 ** (8 * img.nbytes / img.size))
    out = np.zeros(shape=img.shape, dtype=np.uint8)
    out[img > threshold] = level - 1
    return out


def threshold_process_iteration(img: np.ndarray, initial_threshold: int, delta: float) -> np.ndarray:
    assert img.ndim == 2
    level = int(2 ** (8 * img.nbytes / img.size))
    threshold = initial_threshold
    while True:
        m1 = img[img > threshold].mean()
        m2 = img[img <= threshold].mean()
        threshold_new = (m1 + m2) / 2
        if np.abs(threshold_new - threshold) < delta:
            break
        else:
            threshold = threshold_new
    return threshold_process_global(img, threshold)


def threshold_process_otsu(img: np.ndarray) -> Tuple[np.ndarray, float]:
    assert img.ndim == 2
    level = int(2 ** (8 * img.nbytes / img.size))
    # 计算归一化直方图
    histogram = np.zeros(shape=(level,), dtype=np.float)
    elements, counts = np.unique(img, return_counts=True)
    histogram[elements] = counts
    histogram /= histogram.sum()
    p1_k = histogram.cumsum()
    m_k = (np.arange(0, level) * histogram).cumsum()
    m_G = (np.arange(0, level) * histogram).sum()
    sigma_square_B = (m_G * p1_k - m_k) ** 2 / (p1_k * (1 - p1_k))  # 类间方差
    np.nan_to_num(sigma_square_B, nan=0, copy=False)
    k_star = int(np.mean(np.where(sigma_square_B == sigma_square_B.max())).round())
    eta_star = sigma_square_B[k_star] / sigma_square_B.sum()
    return threshold_process_global(img, k_star), eta_star


def threshold_process_multi_block(img: np.ndarray, split: Tuple[int, int]) -> np.ndarray:
    assert img.ndim == 2
    split_x, split_y = split
    out = np.empty_like(img)
    for i in range(split_x):
        for j in range(split_y):
            start_x = i * img.shape[0] // split_x
            end_x = (i + 1) * img.shape[0] // split_x
            start_y = j * img.shape[1] // split_y
            end_y = (j + 1) * img.shape[1] // split_y
            block_otsu, _ = threshold_process_otsu(img[start_x:end_x, start_y:end_y])
            out[start_x:end_x, start_y:end_y] = block_otsu
    return out


if __name__ == '__main__':
    airport = cv2.imread('img/Fig1034(a)(marion_airport).tif', flags=cv2.IMREAD_GRAYSCALE)
    fingerprint = cv2.imread('img/Fig1038(a)(noisy_fingerprint).tif', flags=cv2.IMREAD_GRAYSCALE)
    polymersomes = cv2.imread('img/Fig1039(a)(polymersomes).tif', flags=cv2.IMREAD_GRAYSCALE)
    septagon = cv2.imread('img/Fig1040(a)(large_septagon_gaussian_noise_mean_0_std_50_added).tif',
                          flags=cv2.IMREAD_GRAYSCALE)
    septagon_shaded = cv2.imread('img/Fig1046(a)(septagon_noisy_shaded).tif', flags=cv2.IMREAD_GRAYSCALE)
    print('===== 霍夫变换检测直线 =====')
    level = int(2 ** (8 * airport.nbytes / airport.size))
    airport_canny = debinarize(canny(airport, kernel_size=(13, 13), sigma=2, TL=0.05, TH=0.15), level)
    cv2.imwrite('result/airport_canny.png', airport_canny)
    airport_hough = hough_transform(airport, kernel_size=(13, 13), sigma=2, TL=0.05, TH=0.15)
    tmp = airport_hough / airport_hough.max() * (level - 1)
    tmp = tmp.T
    tmp = tmp[::-1, :]
    tmp = cv2.resize(tmp, (180 * 3, tmp.shape[1]))
    cv2.imwrite('result/airport_hough.png', tmp)
    # 检测直线, 先取小窗，取前两大亮点，画出代表的直线
    # window = airport_hough[175:, 270:310]
    # order1_xy, order2_xy = np.argpartition(window.flatten(), -2)[-2:]
    # order1_x = order1_xy // window.shape[1] + 175
    # order1_y = order1_xy % window.shape[1] + 270
    # order2_x = order2_xy // window.shape[1] + 175
    # order2_y = order2_xy % window.shape[1] + 270
    # print(order1_x, order1_y)
    # print(order2_x, order2_y)
    # theta_1 = order1_x - 90
    # theta_2 = order2_x - 90
    # ro_1 = airport_hough.shape[1] // 2 - order1_y
    # ro_2 = airport_hough.shape[1] // 2 - order2_y
    # print(theta_1, ro_1, theta_2, ro_2)
    # for i in range(airport.shape[0]):
    #     for j in range(airport.shape[1]):
    #         pass
    # 全局阈值处理
    draw_histogram(fingerprint, plt)
    plt.savefig('result/fingerprint_hist.png')
    fingerprint_iteration = threshold_process_iteration(fingerprint, 128, 0.1)
    cv2.imwrite('result/fingerprint_iteration.png', fingerprint_iteration)
    # Otsu 阈值分割
    plt.close()
    draw_histogram(polymersomes, plt)
    plt.savefig('result/polymersomes_hist.png')
    polymersomes_iteration = threshold_process_iteration(polymersomes, 165, 0.1)
    cv2.imwrite('result/polymersomes_iteration.png', polymersomes_iteration)
    polymersomes_otsu, _ = threshold_process_otsu(polymersomes)
    cv2.imwrite('result/polymersomes_otsu.png', polymersomes_otsu)
    # septagon_shaded 的处理
    # 直方图
    plt.close()
    draw_histogram(septagon_shaded, plt)
    plt.savefig('result/septagon_shaded_hist.png')
    # 全局迭代分隔
    septagon_shaded_iteration = threshold_process_iteration(septagon_shaded, 60, 0.1)
    cv2.imwrite('result/septagon_shaded_iteration.png', septagon_shaded_iteration)
    # otsu
    septagon_shaded_otsu, _ = threshold_process_otsu(septagon_shaded)
    cv2.imwrite('result/septagon_shaded_otsu.png', septagon_shaded_otsu)
    # 分块
    split = (2, 3)
    level = int(2 ** (8 * septagon_shaded.nbytes / septagon_shaded.size))
    septagon_shaded_split = septagon_shaded.copy()
    for i in range(1, split[0]):
        septagon_shaded_split[i * septagon_shaded.shape[0] // split[0], :] = level - 1
    for j in range(1, split[1]):
        septagon_shaded_split[:, j * septagon_shaded.shape[1] // split[1]] = level - 1
    cv2.imwrite('result/septagon_shaded_split.png', septagon_shaded_split)
    septagon_shaded_multi_block = threshold_process_multi_block(septagon_shaded, (2, 3))
    cv2.imwrite('result/septagon_shaded_multi_block.png', septagon_shaded_multi_block)
