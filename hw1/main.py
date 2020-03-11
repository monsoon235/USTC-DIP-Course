import os

import numpy as np
from PIL import Image


def open_img(path: str) -> np.ndarray:
    img = Image.open(path)
    return np.array(img)


def save_img(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)


def interpolation_nearest_neighbor(src: np.ndarray, scale: float) -> np.ndarray:
    dst_shape = (round(src.shape[0] * scale), round(src.shape[1] * scale)) + src.shape[2:]
    dst_x = np.arange(0, dst_shape[0])
    dst_y = np.arange(0, dst_shape[1])
    src_x = (dst_x + 0.5) / scale - 0.5
    src_y = (dst_y + 0.5) / scale - 0.5
    src_x = src_x.round().astype(int)
    src_y = src_y.round().astype(int)
    return src[np.ix_(src_x, src_y)]


def interpolation_bilinear(src: np.ndarray, scale: float) -> np.ndarray:
    dst_shape = (round(src.shape[0] * scale), round(src.shape[1] * scale)) + src.shape[2:]
    dst_x = np.arange(0, dst_shape[0])
    dst_y = np.arange(0, dst_shape[1])
    src_x = (dst_x + 0.5) / scale - 0.5
    src_y = (dst_y + 0.5) / scale - 0.5
    # 四个最近邻点，缩小时会出现 src 坐标超出范围，用 max min 约束
    src_x_0 = np.maximum(np.floor(src_x), 0).astype(int)
    src_x_1 = np.minimum(np.ceil(src_x), src.shape[0] - 1).astype(int)
    src_y_0 = np.maximum(np.floor(src_y), 0).astype(int)
    src_y_1 = np.minimum(np.ceil(src_y), src.shape[1] - 1).astype(int)
    pixel_00 = src[np.ix_(src_x_0, src_y_0)]
    pixel_01 = src[np.ix_(src_x_0, src_y_1)]
    pixel_10 = src[np.ix_(src_x_1, src_y_0)]
    pixel_11 = src[np.ix_(src_x_1, src_y_1)]
    delta_x = src_x - src_x_0
    delta_y = src_y - src_y_0
    delta_x, delta_y = np.ix_(delta_x, delta_y)
    # 同时支持单通道和多通道图片
    s = dst_shape[:2] + (1,) * (len(dst_shape) - 2)
    dst = ((1 - delta_x) * (1 - delta_y)).reshape(s) * pixel_00 \
          + ((1 - delta_x) * delta_y).reshape(s) * pixel_01 \
          + (delta_x * (1 - delta_y)).reshape(s) * pixel_10 \
          + (delta_x * delta_y).reshape(s) * pixel_11
    return dst.astype(src.dtype)


def get_img_grad(img_pad: np.ndarray) -> np.ndarray:
    def p(offset_x: int, offset_y) -> np.ndarray:
        assert -1 <= offset_x <= 2
        assert -1 <= offset_y <= 2
        if offset_x == 2:
            clip_x = img_pad[3:, :]
        else:
            clip_x = img_pad[1 + offset_x: -2 + offset_x, :]
        if offset_y == 2:
            clip_xy = clip_x[:, 3:]
        else:
            clip_xy = clip_x[:, 1 + offset_y:-2 + offset_y]
        assert clip_xy.shape == (img_pad.shape[0] - 3, img_pad.shape[1] - 3) + img_pad.shape[2:]
        return clip_xy

    img_grad = np.array([
        [p(0, 0),
         p(0, 1),
         (p(0, 1) - p(0, -1)) / 2,
         (p(0, 2) - p(0, 0)) / 2],
        [p(1, 0),
         p(1, 1),
         (p(1, 1) - p(1, -1)) / 2,
         (p(1, 2) - p(1, 0)) / 2],
        [(p(1, 0) - p(-1, 0)) / 2,
         (p(1, 1) - p(-1, 1)) / 2,
         (p(1, 1) + p(-1, -1) - p(-1, 1) - p(1, -1)) / 4,
         (p(1, 2) + p(-1, 0) - p(-1, 2) - p(1, 0)) / 4],
        [(p(2, 0) - p(0, 0)) / 2,
         (p(2, 1) - p(0, 1)) / 2,
         (p(2, 1) + p(0, -1) - p(0, 1) - p(2, -1)) / 4,
         (p(2, 2) + p(0, 0) - p(0, 2) - p(2, 0)) / 4],
    ])
    img_grad = img_grad.transpose(tuple(range(2, len(img_grad.shape))) + (0, 1))
    return img_grad


def interpolation_bicubic(src: np.ndarray, scale: float) -> np.ndarray:
    dst_shape = (round(src.shape[0] * scale), round(src.shape[1] * scale)) + src.shape[2:]
    dst_x = np.arange(0, dst_shape[0])
    dst_y = np.arange(0, dst_shape[1])
    src_x = (dst_x + 0.5) / scale - 0.5
    src_y = (dst_y + 0.5) / scale - 0.5
    src_x_0 = np.floor(src_x).astype(int)
    src_y_0 = np.floor(src_y).astype(int)
    delta_x = src_x - src_x_0
    delta_y = src_y - src_y_0
    extra_dim_num = len(img.shape) - 2
    img_pad = np.pad(img, pad_width=[(2, 2), (2, 2)] + [(0, 0)] * extra_dim_num, mode='edge').astype(np.int16)  # 避免溢出
    img_grad = get_img_grad(img_pad)
    img_grad = img_grad[np.ix_(src_x_0 + 1, src_y_0 + 1)]
    B1 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [-3, 3, -2, -1],
        [2, -2, 1, 1]
    ])
    B2 = np.array([
        [1, 0, -3, 2],
        [0, 0, 3, -2],
        [0, 1, -2, 1],
        [0, 0, -1, 1]
    ])
    axes = tuple(range(len(img_grad.shape)))
    tmp = np.tensordot(B1, img_grad, axes=(1, axes[-2])).transpose(
        axes[1:-1] + (0, axes[-1])
    )
    A = np.tensordot(tmp, B2, axes=(axes[-1], 0))
    # 计算插值结果
    x_prarm = np.array([np.ones((dst_shape[0],)), delta_x, delta_x ** 2, delta_x ** 3]).T
    y_param = np.array([np.ones((dst_shape[1],)), delta_y, delta_y ** 2, delta_y ** 3]).T
    xy_param = x_prarm.reshape((dst_shape[0], 1, 4, 1)) * y_param.reshape((1, dst_shape[1], 1, 4))
    result = A * xy_param.reshape(dst_shape[0:2] + (1,) * (len(dst_shape) - 2) + (4, 4))
    result = result.sum(axis=(-1, -2))
    result[result < 0] = 0
    result[result > 255] = 255
    return result.astype(src.dtype)


if __name__ == '__main__':
    for filename in os.listdir('img'):
        img = open_img(os.path.join('img', filename))
        name, ext = os.path.splitext(filename)
        for scale in [0.5, 3]:
            save_img(
                interpolation_nearest_neighbor(img, scale),
                os.path.join('result', f'{name}_{scale}_n{ext}')
            )
            save_img(
                interpolation_bilinear(img, scale),
                os.path.join('result', f'{name}_{scale}_b{ext}')
            )
            save_img(
                interpolation_bicubic(img, scale),
                os.path.join('result', f'{name}_{scale}_c{ext}')
            )
