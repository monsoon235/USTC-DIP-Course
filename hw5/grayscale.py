import cv2
import matplotlib.pyplot as plt
import numpy as np


# 膨胀
def dilate_flat_round(img: np.ndarray, radius: int) -> np.ndarray:
    assert img.ndim == 2
    # 找到结构体的点
    di = []
    dj = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if round(np.sqrt(x ** 2 + y ** 2)) <= radius:
                di.append(x)
                dj.append(y)
    di = np.array(di)
    dj = np.array(dj)
    tmp = np.pad(img, pad_width=[(radius, radius), (radius, radius)], mode='minimum')
    out = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.max(tmp[i + radius - di, j + radius - dj])
    return out


# 腐蚀
def erode_flat_round(img: np.ndarray, radius: int) -> np.ndarray:
    assert img.ndim == 2
    # 找到结构体的点
    di = []
    dj = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if round(np.sqrt(x ** 2 + y ** 2)) <= radius:
                di.append(x)
                dj.append(y)
    di = np.array(di)
    dj = np.array(dj)
    tmp = np.pad(img, pad_width=[(radius, radius), (radius, radius)], mode='maximum')
    out = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.min(tmp[i + radius + di, j + radius + dj])
    return out


def open_operation_flat_round(img: np.ndarray, radius: int) -> np.ndarray:
    return dilate_flat_round(erode_flat_round(img, radius), radius)


def close_operation_flat_round(img: np.ndarray, radius: int) -> np.ndarray:
    return erode_flat_round(dilate_flat_round(img, radius), radius)


# 顶帽变换
def top_hat_transform(img: np.ndarray, radius: int) -> np.ndarray:
    return img - open_operation_flat_round(img, radius)


# 形态学平滑
def smooth(img: np.ndarray, radius: int) -> np.ndarray:
    return close_operation_flat_round(open_operation_flat_round(img, radius), radius)


# 形态学梯度
def grad(img: np.ndarray, radius: int) -> np.ndarray:
    return dilate_flat_round(img, radius) - erode_flat_round(img, radius)


def grayscale_morphology_rice():
    print('===== 使用顶帽变换纠正阴影 =====')
    rice = cv2.imread('img/Fig0940(a)(rice_image_with_intensity_gradient).tif')
    rice = cv2.cvtColor(rice, cv2.COLOR_RGB2GRAY)
    _, rice_bin = cv2.threshold(rice, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite('result/rice_bin.png', rice_bin)
    rice_open = open_operation_flat_round(rice, 40)
    cv2.imwrite('result/rice_open.png', rice_open)
    rice_top_hat = top_hat_transform(rice, 40)
    cv2.imwrite('result/rice_top_hat.png', rice_top_hat)
    _, rice_top_hat_bin = cv2.threshold(rice_top_hat, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite('result/rice_top_hat_bin.png', rice_top_hat_bin)


def grayscale_morphology_dowel():
    print('===== 粒度测定 =====')
    dowel = cv2.imread('img/Fig0941(a)(wood_dowels).tif')
    dowel = cv2.cvtColor(dowel, cv2.COLOR_RGB2GRAY)
    dowel_smooth = smooth(dowel, 5)
    cv2.imwrite('result/wood_dowels_smooth.png', dowel_smooth)
    surfaces = [dowel_smooth.sum(dtype=np.int64)]
    for radius in range(1, 36):
        print(f'radius = {radius}')
        dowel_new = open_operation_flat_round(dowel_smooth, radius)
        s = dowel_new.sum(dtype=np.int64)
        surfaces.append(s)
        if radius in [10, 20, 25, 30]:
            cv2.imwrite(f'result/wood_dowels_open_radius={radius}.png', dowel_new)
    difference = []
    for radius in range(1, 36):
        difference.append(surfaces[radius - 1] - surfaces[radius])
    plt.plot(range(1, 36), difference)
    plt.xlabel('r')
    plt.ylabel('Differences in surface areas')
    plt.savefig('result/differences_in_surface_areas.png')


def grayscale_morphology_blobs():
    print('===== 纹理分割 =====')
    blobs = cv2.imread('img/Fig0943(a)(dark_blobs_on_light_background).tif')
    blobs = cv2.cvtColor(blobs, cv2.COLOR_RGB2GRAY)
    blobs_remove_small = close_operation_flat_round(blobs, 30)
    cv2.imwrite('result/blobs_remove_small.png', blobs_remove_small)
    blobs_fill_gaps = open_operation_flat_round(blobs_remove_small, 60)
    cv2.imwrite('result/blobs_fill_gaps.png', blobs_fill_gaps)
    blobs_edge_grad = grad(blobs_fill_gaps, 2)
    blobs_spilt = blobs + blobs_edge_grad
    cv2.imwrite('result/blobs_split.png', blobs_spilt)


if __name__ == '__main__':
    # grayscale_morphology_rice()
    # grayscale_morphology_dowel()
    grayscale_morphology_blobs()
