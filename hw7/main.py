import os
from typing import List

import cv2
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


if __name__ == '__main__':
    # 读取 train
    imgs_train = [cv2.cvtColor(cv2.imread(os.path.join('train', filename)), cv2.COLOR_RGB2GRAY)
                  for filename in os.listdir('train')]
    level = max(int(2 ** (8 * img.nbytes / img.size)) for img in imgs_train)
    freq = np.zeros(shape=(level,), dtype=np.int)
    for img in imgs_train:
        values, counts = np.unique(img, return_counts=True)
        freq[values] += counts
    freq = {value: count for value, count in enumerate(freq)}
    tree = HuffmanTree(freq)
    imgs_test = [cv2.cvtColor(cv2.imread(os.path.join('test', filename)), cv2.COLOR_RGB2GRAY)
                 for filename in os.listdir('test')]
    for img in imgs_test:
        origin_len = img.nbytes * 8
        huffman_coding_len = 0
        for pixel in img.flatten():
            huffman_coding_len += len(tree.encode(pixel))
        print(origin_len, huffman_coding_len, huffman_coding_len / origin_len * 100)
