# PF-PASCAL_kps_visable
用于PF-PASCAL数据集的kps可视化代码

---
## usage
`test.py`和`optimize.py`是示例，在`optimize.py`中返回pck value的时候，将predict key points返回出来，在`test.py`中接收并保存成csv文件，之后使用`test_pairs.csv`中的ground truth来进行key points的可视化
