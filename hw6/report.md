# 《数字图像处理》第五次编程作业

**Notice:** 数学公式在 Typora 中正常显示，需开启 `Markdown 扩展语法` 中的 `内联公式` （重新启动 Typora 生效）。

实验代码见 `main.py`.

## 霍夫变换

霍夫变换的算法见书上，此处不再赘述

![](result/airport.png)

![](result/airport_canny.png)

![](result/airport_hough.png)

## 阈值分割

![](result/fingerprint_hist.png)

<table>
    <tr>
        <td align="center"><img src="result/fingerprint.png"/> <br> 原图 </td>
        <td align="center"> <img src="result/fingerprint_iteration.png"/> <br> 全局迭代法的结果</td>
    </tr>
</table>


![](result/polymersomes_hist.png)

<table>
    <tr>
        <td align="center"><img src="result/polymersomes.png"/> <br> 原图 </td>
        <td align="center"> <img src="result/polymersomes_iteration.png"/> <br> 全局迭代法的结果</td>
        <td align="center"> <img src="result/polymersomes_otsu.png"/> <br> Otsu 算法的结果</td>
    </tr>
</table>


![](result/septagon_shaded_hist.png)

<table>
    <tr>
        <td align="center"><img src="result/septagon_shaded.png"/> <br> 原图 </td>
        <td align="center"> <img src="result/septagon_shaded_iteration.png"/> <br> 全局迭代法的结果</td>
    </tr>
    <tr>
        <td align="center"><img src="result/septagon_shaded_otsu.png"/> <br> Otsu 算法的结果 </td>
        <td align="center"> <img src="result/septagon_shaded_split.png"/> <br> 分块</td>
        <td align="center"> <img src="result/septagon_shaded_multi_block.png"/> <br> 分块 Otsu 算法的结果</td>
    </tr>
</table>
