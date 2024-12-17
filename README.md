# FastThresholdClustering
`FastThresholdClustering` 是一个基于 FAISS 的高效向量聚类算法，特别适用于大规模向量数据的聚类任务。该算法使用余弦相似度作为距离度量，并支持 GPU 加速。
我来为这个聚类算法写一个详细的接口文档：

# FastThresholdClustering 文档

## 简介
`FastThresholdClustering` 是一个基于 FAISS 的高效向量聚类算法，特别适用于大规模向量数据的聚类任务。该算法使用余弦相似度作为距离度量，并支持 GPU 加速。

## 主要特点
- 支持 GPU 加速
- 自动参数优化
- 内存使用优化
- 性能监控和日志记录
- 批处理处理大规模数据
- 噪声点检测

## 快速开始

```python
from fast_clustering import fast_cluster_embeddings

# 使用便捷函数进行聚类
labels = fast_cluster_embeddings(
    embeddings,
    similarity_threshold=0.8,
    min_samples=5,
    use_gpu=True
)
```

## 详细API

### FastThresholdClustering 类

#### 初始化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| similarity_threshold | float | 0.8 | 聚类相似度阈值，范围[0,1] |
| min_samples | int | 5 | 形成簇所需的最小样本数 |
| use_gpu | bool | True | 是否使用GPU加速 |
| nprobe | int | 8 | FAISS索引的nprobe参数 |
| batch_size | int | 1000 | 批处理大小 |
| n_workers | int | None | 并行处理的工作线程数 |

#### 主要方法

```python
def fit(self, embeddings: np.ndarray) -> FastThresholdClustering:
    """
    对输入的向量进行聚类
    
    参数:
        embeddings: shape为(n_samples, n_features)的numpy数组
        
    返回:
        self: 返回聚类器实例
    """
```

### 便捷函数

```python
def fast_cluster_embeddings(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.8,
    min_samples: int = 5,
    use_gpu: bool = True,
    nprobe: int = None,
    batch_size: int = None,
    n_workers: int = None
) -> np.ndarray:
    """
    快速聚类接口函数
    
    参数:
        embeddings: 输入向量，shape为(n_samples, n_features)
        similarity_threshold: 聚类相似度阈值
        min_samples: 最小样本数
        use_gpu: 是否使用GPU
        nprobe: FAISS索引的nprobe参数（可选）
        batch_size: 批处理大小（可选）
        n_workers: 工作线程数（可选）
    
    返回:
        labels: 聚类标签数组，shape为(n_samples,)
    """
```

## 返回值说明
- 聚类结果存储在 `labels_` 属性中
- 标签为 -1 表示噪声点
- 其他标签从 0 开始连续编号

## 性能监控
算法内置了性能监控系统，会自动记录：
- 各阶段耗时
- 内存使用情况
- 聚类进度
- 最终聚类统计信息

## 使用示例

```python
import numpy as np
from fast_clustering import FastThresholdClustering

# 准备数据
embeddings = np.random.random((10000, 768))

# 创建聚类器
clusterer = FastThresholdClustering(
    similarity_threshold=0.8,
    min_samples=5,
    use_gpu=True
)

# 执行聚类
clusterer.fit(embeddings)

# 获取聚类结果
labels = clusterer.labels_
```

## 注意事项
1. 输入向量会自动进行L2归一化
2. 大规模数据集建议启用GPU加速
3. 参数会根据数据规模自动优化
4. 内存使用会随数据规模增长
