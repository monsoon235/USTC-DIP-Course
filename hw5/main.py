import cv2
import numpy as np
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

DEBUG = True


def open_img(path: str) -> np.ndarray:
    return cv2.imread(path)


def save_img(img: np.ndarray, path: str):
    cv2.imwrite(path, img)


def normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


def draw_histogram(img: np.ndarray, ax):
    level = int(2 ** (8 * img.nbytes / img.size))
    ax.hist(img.flatten(), bins=level, density=True, histtype='stepfilled')


def binarize(img: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.bool)
    out[img >= 128] = True
    return out


def debinarize(img_bin: np.ndarray) -> np.ndarray:
    img = np.zeros_like(img_bin, dtype=np.uint8)
    img[img_bin] = 255
    return img


class StructuralElement:
    def isIn(self, point: np.ndarray) -> bool:
        raise NotImplementedError


# 腐蚀
def corrode(img: np.ndarray, b: np.ndarray) -> np.ndarray:
    pass


# 膨胀，kernel 为十字
def dilate_cross(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 2 and img.dtype == np.bool
    # 加一个边框加速计算
    tmp = np.zeros(shape=(img.shape[0] + 2, img.shape[1] + 2), dtype=np.bool)
    tmp[1:-1, 1:-1] = img
    index_i, index_j = np.where(tmp)
    tmp[index_i - 1, index_j] = True
    tmp[index_i + 1, index_j] = True
    tmp[index_i, index_j - 1] = True
    tmp[index_i, index_j + 1] = True
    return tmp[1:-1, 1:-1]


# 膨胀，kernel 为 51x1
def dilate_51x1(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 2 and img.dtype == np.bool
    # 加一个边框加速计算
    tmp = np.zeros(shape=(img.shape[0] + 50, img.shape[1]), dtype=np.bool)
    tmp[25:-25, :] = img
    index_i, index_j = np.where(tmp)
    for d in range(-25, 26, 1):
        tmp[index_i + d, index_j] = True
    return tmp[25:-25, :]


# 先填充外部，然后取反
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


def top_hat_transform(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 2 and img.dtype == np.uint8


def binary_morphology_exp():
    img = open_img('img/Fig0929(a)(text_image).tif')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_bin = binarize(img)
    print('filling hole...')
    img_bin_fill_hole = fill_hole(img_bin)
    img_out = debinarize(img_bin_fill_hole)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2RGB)
    save_img(img_out, 'result/text_img_fill_hole.png')
    print('extracting long character...')
    img_bin_extract_long_char = extract_long_character(img_bin)
    img_out = debinarize(img_bin_extract_long_char)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2RGB)
    save_img(img_out, 'result/text_img_extract_long_character.png')
    print('clearing boundary...')
    img_bin_clear_boundary = clear_boundary(img_bin)
    img_out = debinarize(img_bin_clear_boundary)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2RGB)
    save_img(img_out, 'result/text_img_clear_boundary.png')


def grayscale_morphology_exp():
    rice = open_img('img/Fig0940(a)(rice_image_with_intensity_gradient).tif')
    wood = open_img('img/Fig0941(a)(wood_dowels).tif')
    blobs = open_img('img/Fig0943(a)(dark_blobs_on_light_background).tif')
    rice = cv2.cvtColor(rice, cv2.COLOR_RGB2GRAY)
    wood = cv2.cvtColor(wood, cv2.COLOR_RGB2GRAY)
    blobs = cv2.cvtColor(blobs, cv2.COLOR_RGB2GRAY)
    print(rice.shape)
    print(wood.shape)
    print(blobs.shape)
    pass


if __name__ == '__main__':
    binary_morphology_exp()
    grayscale_morphology_exp()
