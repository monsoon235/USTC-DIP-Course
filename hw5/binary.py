import cv2
import numpy as np


# 二值化
def binarize(img: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.bool)
    out[img >= 128] = True
    return out


# 重建图像
def debinarize(img_bin: np.ndarray) -> np.ndarray:
    img = np.zeros_like(img_bin, dtype=np.uint8)
    img[img_bin] = 255
    return img


# 膨胀，kernel 为十字
def dilate_cross(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 2 and img.dtype == np.bool
    # 加一个边框加速计算
    tmp = np.pad(img, pad_width=[(1, 1), (1, 1)], mode='constant', constant_values=0)
    index_i, index_j = np.where(tmp)
    tmp[index_i - 1, index_j] = True
    tmp[index_i + 1, index_j] = True
    tmp[index_i, index_j - 1] = True
    tmp[index_i, index_j + 1] = True
    return tmp[1:-1, 1:-1].copy()


# 膨胀，kernel 为 51x1
def dilate_51x1(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 2 and img.dtype == np.bool
    # 加一个边框加速计算
    tmp = np.pad(img, pad_width=[(25, 25), (0, 0)], mode='constant', constant_values=0)
    index_i, index_j = np.where(tmp)
    for d in range(-25, 26, 1):
        tmp[index_i + d, index_j] = True
    return tmp[25:-25, :].copy()


def fill_hole(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 2 and img.dtype == np.bool
    marker = np.zeros(shape=(img.shape[0] + 2, img.shape[1] + 2), dtype=np.bool)
    marker[0, :] = True
    marker[-1, :] = True
    marker[:, 0] = True
    marker[:, -1] = True
    mask = marker.copy()
    mask[1:-1, 1:-1] = ~img
    while True:
        marker_pre = marker
        dilation = dilate_cross(marker)
        marker = dilation & mask
        if np.all(marker_pre == marker):
            break
    return ~marker[1:-1, 1:-1]


def extract_long_character(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 2 and img.dtype == np.bool
    marker = ~dilate_51x1(~img)
    # 通过膨胀重建
    mask = img
    while True:
        marker_pre = marker
        dilation = dilate_cross(marker)
        marker = dilation & mask
        if np.all(marker_pre == marker):
            break
    return marker


def clear_boundary(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 2 and img.dtype == np.bool
    marker = img.copy()
    marker[1:-1, 1:-1] = False
    # 通过膨胀重建
    mask = img
    while True:
        marker_pre = marker
        dilation = dilate_cross(marker)
        marker = dilation & mask
        if np.all(marker_pre == marker):
            break
    return marker ^ img


def binary_morphology_exp():
    img = cv2.imread('img/Fig0929(a)(text_image).tif')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_bin = binarize(img)
    print('filling hole...')
    img_bin_fill_hole = fill_hole(img_bin)
    img_out = debinarize(img_bin_fill_hole)
    cv2.imwrite('result/text_fill_hole.png', img_out)
    print('extracting long character...')
    img_bin_extract_long_char = extract_long_character(img_bin)
    img_out = debinarize(img_bin_extract_long_char)
    cv2.imwrite('result/text_extract_long_character.png', img_out)
    print('clearing boundary...')
    img_bin_clear_boundary = clear_boundary(img_bin)
    img_out = debinarize(img_bin_clear_boundary)
    cv2.imwrite('result/text_clear_boundary.png', img_out)


if __name__ == '__main__':
    binary_morphology_exp()
