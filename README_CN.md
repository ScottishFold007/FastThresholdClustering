# FastThresholdClustering


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


# FastThresholdClustering 参数详解

## 核心参数

### similarity_threshold
| 属性 | 说明 |
|------|------|
| 类型 | float |
| 默认值 | 0.8 |
| 取值范围 | [0, 1] |
| 功能 | 判定两个向量是否属于同一簇的相似度阈值 |

**详细说明**：
- 值越大，聚类标准越严格，形成的簇越多
- 值越小，聚类越宽松，簇的数量越少
- 建议取值：
  - 0.7-0.8：适用于一般文本向量
  - 0.8-0.9：适用于需要高精度匹配的场景
  - >0.9：极其严格的匹配要求
- 对性能影响：阈值越高，计算速度越快

### min_samples
| 属性 | 说明 |
|------|------|
| 类型 | int |
| 默认值 | 5 |
| 取值范围 | >= 2 |
| 功能 | 形成一个有效簇所需的最小样本数量 |

**详细说明**：
- 小于此数量的簇会被标记为噪声点（标签为-1）
- 参数设置建议：
  - 小数据集（<1000）：2-5
  - 中等数据集（1000-10000）：5-10
  - 大数据集（>10000）：10-20
- 影响噪声点判定的关键参数
- 值越大，噪声点越多，簇的质量越高

## 性能相关参数

### use_gpu
| 属性 | 说明 |
|------|------|
| 类型 | bool |
| 默认值 | True |
| 功能 | 是否使用GPU加速计算 |

**详细说明**：
- True：使用GPU加速计算
- False：使用CPU计算
- 性能影响：
  - GPU模式：适合大规模数据（>10万条）
  - CPU模式：适合小规模数据（<10万条）
- 内存使用：
  - GPU模式受显存限制
  - CPU模式受内存限制
- 建议：有GPU时优先使用GPU模式

### nprobe
| 属性 | 说明 |
|------|------|
| 类型 | int |
| 默认值 | 8 |
| 取值范围 | [1, nlist] |
| 功能 | FAISS索引搜索时访问的聚类单元数量 |

**详细说明**：
- 影响搜索精度和速度的平衡参数
- 建议取值：
  - 小数据集（<10k）：4-8
  - 中等数据集（10k-100k）：8-16
  - 大数据集（>100k）：16-32
- 值越大：
  - 优点：搜索结果越准确
  - 缺点：搜索速度越慢
- 会根据数据规模自动调整

### batch_size
| 属性 | 说明 |
|------|------|
| 类型 | int |
| 默认值 | 1000 |
| 取值范围 | [100, 数据集大小] |
| 功能 | 批处理大小，影响内存使用和计算速度 |

**详细说明**：
- 建议取值：
  - GPU模式：500-2000
  - CPU模式：200-1000
- 内存影响：
  - 值越大，内存使用越多
  - 值越小，计算时间越长
- 自动调整：
  - 小数据集：较小batch_size
  - 大数据集：较大batch_size
- 需要根据可用内存调整

### n_workers
| 属性 | 说明 |
|------|------|
| 类型 | int |
| 默认值 | None |
| 取值范围 | [1, CPU核心数] |
| 功能 | 并行处理的工作线程数 |

**详细说明**：
- None时自动设置为 min(CPU核心数, 8)
- 建议取值：
  - 小数据集：2-4线程
  - 中等数据集：4-8线程
  - 大数据集：8-16线程
- 注意事项：
  - 线程数过多可能导致资源竞争
  - 需要考虑系统其他进程的资源需求
  - GPU模式下影响较小

## 参数组合建议

### 小数据集优化（<10k样本）
```python
FastThresholdClustering(
    similarity_threshold=0.75,
    min_samples=3,
    use_gpu=False,
    nprobe=4,
    batch_size=500,
    n_workers=4
)
```

### 大数据集优化（>100k样本）
```python
FastThresholdClustering(
    similarity_threshold=0.85,
    min_samples=10,
    use_gpu=True,
    nprobe=32,
    batch_size=2000,
    n_workers=8
)
```

### 高精度要求场景
```python
FastThresholdClustering(
    similarity_threshold=0.9,
    min_samples=5,
    use_gpu=True,
    nprobe=64,
    batch_size=1000,
    n_workers=8
)
```

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

## 详细使用用例见**example.py**

## 注意事项
1. 输入向量会自动进行L2归一化
2. 大规模数据集建议启用GPU加速
3. 参数会根据数据规模自动优化
4. 内存使用会随数据规模增长
