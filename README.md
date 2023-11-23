# DV-SGSC: Dual-view superpixel graph subspace clustering for hyperspectral remote sensing images
## Abstract
> Hyperspectral remote sensing has become a powerful technology for observing the geoscience of the Earth surface. The unique structure of hyperspectral imagery (HSI) provides a wealth of information; however, the inaccessibility of labels in image classification limits its application. Although the unsupervised subspace clustering could effectively utilize the high dimensionality characteristic of HSI without labels, present methods still suffer some limitations. Firstly, the memory consumption of subspace clustering expands dramatically with increasing data volume, inhibiting its applicability to large datasets. Secondly, existing studies only employ superpixel segmentation techniques to reduce the amount of data in the preprocessing stage of HSI, without thoroughly exploring and exploiting the inherent prior information contained within superpixels, resulting in inferior clustering performance. To address these issues, we propose a dual-view superpixel graph subspace clustering (DV-SGSC) method that integrates the intrinsic graph structure of the superpixel spatial domain and the spectral domain. Superpixel segmentation greatly reduces the amount of data without sacrificing accuracy, thereby alleviating the memory consumption problem caused by large datasets. We also designed a logical, intuitive method for creating an intrinsic graph of superpixels in both the spatial and spectral domains, and evaluated its efficacy. Finally, we present a joint optimization framework that can simultaneously process spatial and spectral superpixel information and obtain a unified self-expressive matrix. Experiments using four hyperspectral benchmark datasets demonstrate the effectiveness and superiority of the proposed method.
## Architecture
![](https://github.com/HK-code/DV-SGSC/blob/main/images/flowchart.jpg)
## Datasets
SalinasA, Pavia university and Pavia center datasets you can download from [here](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University), and WHU_Hi_LongKou dataset you can download from [here](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm).
## Evaluation
```python
python run.py
```
## Generate superpixels
If you want to use ERS to generate superpixels, first reduce the dimensionality of the hyperspectral image and then use the ERS algorithm to segment it.
```python
run ERS/main.py
run ERS/makesuperpixel.m
```
## More
If you have any questions and needs, you can contact me, my email is: huangkun@whu.edu.cn.
