# FastThresholdClustering
### [中文文档](README_CN.md)

## Introduction
`FastThresholdClustering` is an efficient vector clustering algorithm based on FAISS, particularly suitable for large-scale vector data clustering tasks. The algorithm uses cosine similarity as the distance metric and supports GPU acceleration.

## Key Features
- GPU acceleration support
- Automatic parameter optimization
- Memory usage optimization
- Performance monitoring and logging
- Batch processing for large-scale data
- Noise point detection

## Quick Start

```python
from fast_clustering import fast_cluster_embeddings

# Use the convenience function for clustering
labels = fast_cluster_embeddings(
    embeddings,
    similarity_threshold=0.8,
    min_samples=5,
    use_gpu=True
)
```

# FastThresholdClustering Parameter Details

## Core Parameters

### similarity_threshold
| Property | Description |
|----------|-------------|
| Type | float |
| Default | 0.8 |
| Range | [0, 1] |
| Function | Similarity threshold for determining if two vectors belong to the same cluster |

**Detailed Description**:
- Higher values lead to stricter clustering and more clusters
- Lower values result in looser clustering and fewer clusters
- Recommended values:
  - 0.7-0.8: Suitable for general text vectors
  - 0.8-0.9: Suitable for high-precision matching scenarios
  - >0.9: Extremely strict matching requirements
- Performance impact: Higher threshold leads to faster computation

### min_samples
| Property | Description |
|----------|-------------|
| Type | int |
| Default | 5 |
| Range | >= 2 |
| Function | Minimum number of samples required to form a valid cluster |

**Detailed Description**:
- Clusters with fewer samples are marked as noise points (label -1)
- Parameter setting recommendations:
  - Small datasets (<1000): 2-5
  - Medium datasets (1000-10000): 5-10
  - Large datasets (>10000): 10-20
- Key parameter affecting noise point determination
- Higher values result in more noise points and higher cluster quality

## Performance Parameters

### use_gpu
| Property | Description |
|----------|-------------|
| Type | bool |
| Default | True |
| Function | Whether to use GPU acceleration |

**Detailed Description**:
- True: Use GPU acceleration
- False: Use CPU computation
- Performance impact:
  - GPU mode: Suitable for large-scale data (>100k entries)
  - CPU mode: Suitable for small-scale data (<100k entries)
- Memory usage:
  - GPU mode limited by VRAM
  - CPU mode limited by RAM
- Recommendation: Prefer GPU mode when available

### nprobe
| Property | Description |
|----------|-------------|
| Type | int |
| Default | 8 |
| Range | [1, nlist] |
| Function | Number of cluster units to visit during FAISS index search |

**Detailed Description**:
- Balances search accuracy and speed
- Recommended values:
  - Small datasets (<10k): 4-8
  - Medium datasets (10k-100k): 8-16
  - Large datasets (>100k): 16-32
- Higher values:
  - Pros: More accurate search results
  - Cons: Slower search speed
- Automatically adjusted based on data scale

### batch_size
| Property | Description |
|----------|-------------|
| Type | int |
| Default | 1000 |
| Range | [100, dataset size] |
| Function | Batch size affecting memory usage and computation speed |

**Detailed Description**:
- Recommended values:
  - GPU mode: 500-2000
  - CPU mode: 200-1000
- Memory impact:
  - Larger values use more memory
  - Smaller values increase computation time
- Auto-adjustment:
  - Small datasets: Smaller batch_size
  - Large datasets: Larger batch_size
- Adjust based on available memory

### n_workers
| Property | Description |
|----------|-------------|
| Type | int |
| Default | None |
| Range | [1, CPU cores] |
| Function | Number of parallel processing worker threads |

**Detailed Description**:
- When None, automatically set to min(CPU cores, 8)
- Recommended values:
  - Small datasets: 2-4 threads
  - Medium datasets: 4-8 threads
  - Large datasets: 8-16 threads
- Considerations:
  - Too many threads may cause resource contention
  - Consider resource needs of other system processes
  - Less impact in GPU mode

## Parameter Combination Recommendations

### Small Dataset Optimization (<10k samples)
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

### Large Dataset Optimization (>100k samples)
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

### High Precision Scenarios
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

#### Main Methods

```python
def fit(self, embeddings: np.ndarray) -> FastThresholdClustering:
    """
    Perform clustering on input vectors
    
    Parameters:
        embeddings: numpy array with shape (n_samples, n_features)
        
    Returns:
        self: Returns the clustering instance
    """
```

### Convenience Function

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
    Quick clustering interface function
    
    Parameters:
        embeddings: Input vectors with shape (n_samples, n_features)
        similarity_threshold: Clustering similarity threshold
        min_samples: Minimum sample count
        use_gpu: Whether to use GPU
        nprobe: FAISS index nprobe parameter (optional)
        batch_size: Batch size (optional)
        n_workers: Number of worker threads (optional)
    
    Returns:
        labels: Clustering label array with shape (n_samples,)
    """
```

## Return Value Description
- Clustering results stored in `labels_` attribute
- Label -1 indicates noise points
- Other labels numbered consecutively from 0

## Performance Monitoring
The algorithm includes a built-in performance monitoring system that automatically records:
- Time spent in each phase
- Memory usage
- Clustering progress
- Final clustering statistics

## Usage Example

```python
import numpy as np
from fast_clustering import FastThresholdClustering

# Prepare data
embeddings = np.random.random((10000, 768))

# Create clusterer
clusterer = FastThresholdClustering(
    similarity_threshold=0.8,
    min_samples=5,
    use_gpu=True
)

# Perform clustering
clusterer.fit(embeddings)

# Get clustering results
labels = clusterer.labels_
```

## See **example.py** for detailed usage examples

## Notes
1. Input vectors are automatically L2 normalized
2. GPU acceleration recommended for large-scale datasets
3. Parameters are automatically optimized based on data scale
4. Memory usage increases with data scale
