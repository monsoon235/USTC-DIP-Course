import cv2
import matplotlib.pylab as plt
import numpy as np


######################################## 二值形态学 ########################################3

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
    img_bin_fill_hole = fill_hole(img_bin)
    img_out = debinarize(img_bin_fill_hole, 256)
    print('孔洞填充 => text_fill_hole.png')
    cv2.imwrite('result/text_fill_hole.png', img_out)
    img_bin_extract_long_char = extract_long_character(img_bin)
    img_out = debinarize(img_bin_extract_long_char, 256)
    print('长字符提取 => text_extract_long_character.png')
    cv2.imwrite('result/text_extract_long_character.png', img_out)
    img_bin_clear_boundary = clear_boundary(img_bin)
    img_out = debinarize(img_bin_clear_boundary, 256)
    print('清除边界 => text_clear_boundary.png')
    cv2.imwrite('result/text_clear_boundary.png', img_out)


#################################### 灰度形态学 ########################################3

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
    img_pad = np.pad(img, pad_width=[(radius, radius), (radius, radius)], mode='minimum')
    # operation_field = []
    # for index in range(di.size):
    #     operation_field.append(img_pad[di[index] + radius:di[index] + radius + img.shape[0],
    #                            dj[index] + radius:dj[index] + radius + img.shape[1]])
    # out = np.max(operation_field, axis=2)
    out = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.max(img_pad[i + radius - di, j + radius - dj])
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
    img_pad = np.pad(img, pad_width=[(radius, radius), (radius, radius)], mode='maximum')
    # operation_field = []
    # for index in range(di.size):
    #     operation_field.append(img_pad[di[index] + radius:di[index] + radius + img.shape[0],
    #                            dj[index] + radius:dj[index] + radius + img.shape[1]])
    # out = np.max(operation_field, axis=2)
    out = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.min(img_pad[i + radius + di, j + radius + dj])
    return out


# 开操作
def open_operation_flat_round(img: np.ndarray, radius: int) -> np.ndarray:
    return dilate_flat_round(erode_flat_round(img, radius), radius)


# 闭操作
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


# 纠正阴影
def correct_the_shadow_rice():
    print('===== 纠正阴影 =====')
    rice = cv2.imread('img/Fig0940(a)(rice_image_with_intensity_gradient).tif')
    rice = cv2.cvtColor(rice, cv2.COLOR_RGB2GRAY)
    _, rice_otsu = cv2.threshold(rice, 0, 255, cv2.THRESH_OTSU)
    print('Otsu 算法二值化 => rice_otsu.png')
    cv2.imwrite('result/rice_otsu.png', rice_otsu)
    rice_open = open_operation_flat_round(rice, 40)
    print('开操作 => rice_open.png')
    cv2.imwrite('result/rice_open.png', rice_open)
    rice_top_hat = top_hat_transform(rice, 40)
    print('顶帽变换 => rice_tophat.png')
    cv2.imwrite('result/rice_top_hat.png', rice_top_hat)
    _, rice_top_hat_otsu = cv2.threshold(rice_top_hat, 0, 255, cv2.THRESH_OTSU)
    print('顶帽变换后 Otsu 算法二值化 => rice_top_hat_otsu.png')
    cv2.imwrite('result/rice_top_hat_otsu.png', rice_top_hat_otsu)


# 粒度测定
def determinate_granularity_dowel():
    print('===== 粒度测定 =====')
    dowel = cv2.imread('img/Fig0941(a)(wood_dowels).tif')
    dowel = cv2.cvtColor(dowel, cv2.COLOR_RGB2GRAY)
    dowel_smooth = smooth(dowel, 5)
    print('形态学平滑 => dowels_smooth.png')
    cv2.imwrite('result/dowels_smooth.png', dowel_smooth)
    surfaces = [dowel_smooth.sum(dtype=np.int64)]
    print('根据不同的半径进行开操作 ...')
    for radius in range(1, 36):
        print(f'\tr={radius}')
        dowel_new = open_operation_flat_round(dowel_smooth, radius)
        s = dowel_new.sum(dtype=np.int64)
        surfaces.append(s)
        if radius in [10, 20, 25, 30]:
            cv2.imwrite(f'result/dowels_open_r={radius}.png', dowel_new)
    difference = []
    for radius in range(1, 36):
        difference.append(surfaces[radius - 1] - surfaces[radius])
    plt.plot(range(1, 36), difference)
    plt.xlabel('r')
    plt.ylabel('Differences in surface areas')
    plt.xlim(0, 35)
    plt.ylim(bottom=0)
    print('surface areas 差值 => dowels_differences_in_surface_areas.png')
    plt.savefig('result/dowels_differences_in_surface_areas.png')


# 纹理分割
def spilt_texture_blobs():
    print('===== 纹理分割 =====')
    blobs = cv2.imread('img/Fig0943(a)(dark_blobs_on_light_background).tif')
    blobs = cv2.cvtColor(blobs, cv2.COLOR_RGB2GRAY)
    blobs_remove_small = close_operation_flat_round(blobs, 30)
    print('用闭操作移除小尺寸物件 => blobs_remove_small.png')
    cv2.imwrite('result/blobs_remove_small.png', blobs_remove_small)
    blobs_fill_gaps = open_operation_flat_round(blobs_remove_small, 60)
    print('用开操作连接大尺寸物件 => blobs_fill_gaps.png')
    cv2.imwrite('result/blobs_fill_gaps.png', blobs_fill_gaps)
    blobs_edge_grad = grad(blobs_fill_gaps, 2)
    blobs_spilt = blobs + blobs_edge_grad
    print('分隔结果 => blobs_split.png')
    cv2.imwrite('result/blobs_split.png', blobs_spilt)


def grayscale_morphology_exp():
    correct_the_shadow_rice()
    determinate_granularity_dowel()
    spilt_texture_blobs()


#########################################################################

if __name__ == '__main__':
    binary_morphology_exp()
    grayscale_morphology_exp()
