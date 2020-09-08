import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Node:

    def __init__(self, name, value, left, right, parent=None) -> None:
        self.name = name
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent

    def __str__(self) -> str:
        # return f'{self.value}'
        return f'(name={self.name}, value={self.value}, left={self.left}, right={self.right})'


class HuffmanTree:

    def __init__(self, freq: dict) -> None:
        ap = sorted(freq.items(), key=lambda item: item[1])
        leaves = [Node(item[0], item[1], None, None) for item in ap]
        nodes = leaves.copy()
        while len(nodes) >= 2:
            left = nodes.pop(0)
            right = nodes.pop(0)
            new = Node(None, value=left.value + right.value, left=left, right=right)
            left.parent = new
            right.parent = new
            start = 0
            end = len(nodes)
            while start != end:
                mid = (start + end) // 2
                if new.value < nodes[mid].value:
                    end = mid
                elif new.value > nodes[mid].value:
                    start = mid + 1
                else:
                    start = mid
                    end = mid
            # start 是写入位置
            nodes.insert(start, new)
        self.root = nodes[0]
        self.coding = {}
        for leaf in leaves:
            assert leaf.value not in self.coding
            buffer = []
            now = leaf
            while now.parent is not None:
                if now is now.parent.left:
                    buffer.insert(0, 0)
                else:
                    buffer.insert(0, 1)
                now = now.parent
            self.coding[leaf.name] = buffer

    def encode(self, a) -> List[int]:
        assert a in self.coding
        return self.coding[a]

    def decode(self, code: List[int]):
        now = self.root
        for b in code:
            assert now is not None
            if b == 0:
                now = now.left
            else:
                now = now.right
        assert now.left is None and now.right is None
        return now.name


# 画直方图
def draw_histogram(img: np.ndarray, ax):
    level = int(2 ** (8 * img.nbytes / img.size))
    ax.hist(img.flatten(), bins=level, range=[0, level - 1], density=True, histtype='stepfilled')
    ax.xlim(0, level - 1)
    ax.ylim(bottom=0)


if __name__ == '__main__':
    imgs_train = [cv2.cvtColor(cv2.imread(os.path.join('train', filename)), cv2.COLOR_RGB2GRAY)
                  for filename in os.listdir('train')]
    level = max(int(2 ** (8 * img.nbytes / img.size)) for img in imgs_train)
    freq = np.zeros(shape=(level,), dtype=np.int)
    for img in imgs_train:
        values, counts = np.unique(img, return_counts=True)
        freq[values] += counts
    # 画出频率直方图
    plt.bar(range(freq.size), freq / freq.sum(), width=1)
    plt.title('hist of all train images')
    plt.xlim(0, level - 1)
    plt.ylim(bottom=0)
    plt.savefig('result/train_hist.png')
    plt.close()
    # 构建树
    freq = {value: count for value, count in enumerate(freq)}
    tree = HuffmanTree(freq)
    print('编码为：')
    print('\n'.join(
        f"\t{k}:\t{''.join(map(str, v))}"
        for k, v in sorted(tree.coding.items(), key=lambda item: item[0])
    ))
    print()
    # 测试
    print('文件名\t原大小 (KB)\t压缩后大小 (KB)\t压缩率 (%)')
    for filename in os.listdir('test'):
        img = cv2.cvtColor(cv2.imread(os.path.join('test', filename)), cv2.COLOR_RGB2GRAY)
        # 画直方图
        draw_histogram(img, plt)
        plt.savefig(f'result/{os.path.splitext(filename)[0]}_hist.png')
        plt.close()
        origin_len = img.nbytes * 8
        huffman_len = 0
        for pixel in img.flatten():
            huffman_len += len(tree.encode(pixel))
        origin_len /= 8 * 1024
        huffman_len /= 8 * 1024
        print(f'{filename}\t{origin_len}\t{huffman_len}\t{huffman_len / origin_len * 100}')
