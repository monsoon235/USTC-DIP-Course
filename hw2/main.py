import os
import typing

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def open_img(path: str) -> np.ndarray:
    img = Image.open(path)
    return np.array(img)


def save_img(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)


def count_pixel_level(img: np.ndarray) -> typing.Tuple[np.ndarray, int]:
    level = int(2 ** (8 * img.nbytes / img.size))
    pr = np.zeros(shape=(level,), dtype=int)
    elements, counts = np.unique(img, return_counts=True)
    pr[elements] = counts  # 统计每个灰度值的个数
    return pr, level


def draw_histogram(img: np.ndarray, ax):
    level = int(2 ** (8 * img.nbytes / img.size))
    ax.hist(img.flatten(), bins=level, density=True, histtype='stepfilled')


def histogram_equalize(img: np.ndarray, step: int) -> np.ndarray:
    assert len(img.shape) == 2
    pr, level = count_pixel_level(img)
    # 映射到 [0,255] 整数区间
    sk = pr.cumsum() / img.size
    # 映射到 [-0.5, step-0.5] 的区间，再取 round
    sk = sk * step - 0.5
    sk = sk.round()
    sk[sk == step] = step - 1
    # 映射回 [0,255] 整数区间
    sk *= (level - 1) / (step - 1)
    return sk[img].round().astype(np.uint8)


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


def mean_filter(img: np.ndarray, n: int = 1) -> np.ndarray:
    for i in range(n):
        img = np.pad(img, [(1, 1), (1, 1)] + [(0, 0)] * (len(img.shape) - 2), mode='edge')
        img_im2col = im2col(img, kernel_size=(3, 3), stride=(1, 1))
        img = img_im2col.mean(axis=(2, 3))
    return img.round().astype(np.uint8)


def median_filter(img: np.ndarray, n: int = 1) -> np.ndarray:
    extra_dim_num = img.ndim - 2
    for i in range(n):
        img_pad = np.pad(img, [(1, 1), (1, 1)] + [(0, 0)] * extra_dim_num, mode='edge')
        img2col = im2col(img_pad, kernel_size=(3, 3), stride=(1, 1))
        img = np.median(img2col, axis=(2, 3))
    return img.round().astype(np.uint8)


def sharpen(img: np.ndarray) -> np.ndarray:
    extra_dim_num = img.ndim - 2
    operand = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ]).reshape((1, 1, 3, 3) + (1,) * extra_dim_num)
    img_pad = np.pad(img, [(1, 1), (1, 1)] + [(0, 0)] * extra_dim_num, mode='edge')
    img2col = im2col(img_pad, kernel_size=(3, 3), stride=(1, 1))
    img = img + (img2col.astype(np.int16) * operand).sum(axis=(2, 3))
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


if __name__ == '__main__':
    img = open_img('img/bridge.jpg')
    name, ext = os.path.splitext('bridge.jpg')
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Origin')
    draw_histogram(img, axes[0, 1])
    for i, n in enumerate([2, 64, 256]):
        img_hist_eq = histogram_equalize(img, n)
        axes[i + 1, 0].imshow(img_hist_eq, cmap='gray')
        axes[i + 1, 0].set_title(f'n={n}')
        draw_histogram(img_hist_eq, axes[i + 1, 1])
        save_img(
            img_hist_eq,
            os.path.join('result', f'{name}_hist_eq_{n}{ext}')
        )
    fig.tight_layout()
    fig.savefig('pic/bridge_show.jpg')

    img = open_img('img/circuit.jpg')
    name, ext = os.path.splitext('circuit.jpg')
    print('waiting filtering...')
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.reshape(2, 10)
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Origin')
    axes[1, 0].imshow(img, cmap='gray')
    axes[1, 0].set_title('Origin')
    for i, n in enumerate([1, 2, 5, 10, 50, 100, 500, 1000, 10000]):
        img_mean = mean_filter(img, n=n)
        img_median = median_filter(img, n=n)
        axes[0, i + 1].imshow(img_mean, cmap='gray')
        axes[0, i + 1].set_title(f'n={n}')
        axes[1, i + 1].imshow(img_median, cmap='gray')
        axes[1, i + 1].set_title(f'n={n}')
        save_img(
            img_mean,
            os.path.join('result', f'{name}_mean_filtering_{n}{ext}')
        )
        save_img(
            img_median,
            os.path.join('result', f'{name}_median_filtering_{n}{ext}')
        )
    fig.tight_layout()
    fig.savefig('pic/circuit_show.jpg')

    img = open_img('img/moon.jpg')
    name, ext = os.path.splitext('moon.jpg')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    img_sharpen = sharpen(img)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Origin')
    axes[1].imshow(img_sharpen, cmap='gray')
    save_img(
        img_sharpen,
        os.path.join('result', f'{name}_sharpen{ext}')
    )
    fig.tight_layout()
    fig.savefig('pic/moon_show.jpg')
