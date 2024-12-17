import os
import gc
import time
import psutil
import faiss
import logging
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
from typing import Optional
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial


# 添加日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Timer:
    """计时器类，用于记录各阶段耗时"""
    def __init__(self):
        self.times = {}
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.time()

    def stop(self, name):
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0

    def get_time(self, name):
        return self.times.get(name, 0)

    def summary(self):
        logging.info("\n=== 性能统计 ===")
        total_time = sum(self.times.values())
        for name, elapsed in sorted(self.times.items(), key=lambda x: x[1], reverse=True):
            percentage = (elapsed / total_time) * 100
            logging.info(f"{name}: {elapsed:.2f}秒 ({percentage:.1f}%)")
        logging.info(f"总耗时: {total_time:.2f}秒")

@contextmanager
def timer_context(timer, name):
    try:
        start_time = time.time()
        yield
    finally:
        elapsed = time.time() - start_time
        timer.times[name] = elapsed

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_step(message: str, timer: Timer):
    memory_usage = get_memory_usage()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"[{timestamp}] {message}")
    logging.info(f"内存使用: {memory_usage:.2f} MB")

def get_recommended_nprobe(n_samples: int) -> int:
    """根据数据规模推荐nprobe值"""
    if n_samples < 10000:
        return 4  # 小数据集，追求速度
    elif n_samples < 100000:
        return 8  # 中等数据集，平衡速度和准确性
    elif n_samples < 1000000:
        return 16  # 大数据集，稍微偏向准确性
    else:
        return 32  # 超大数据集，重视准确性

def get_recommended_params(n_samples: int, d: int):
    """获取推荐参数"""
    params = {
        'nlist': min(int(np.sqrt(n_samples) * 2), n_samples // 20),
        'nprobe': get_recommended_nprobe(n_samples),  # 使用新的nprobe策略
        'batch_size': min(1000, n_samples // 10),
        'n_workers': min(os.cpu_count(), 8)
    }
    return params

class FastThresholdClustering:
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_samples: int = 5,
        use_gpu: bool = True,
        nprobe: int = 8,
        batch_size: int = 1000,
        n_workers: int = None
    ):
        self.timer = Timer()
        with timer_context(self.timer, "初始化"):
            log_step("初始化聚类器", self.timer)
            self.similarity_threshold = similarity_threshold
            self.min_samples = min_samples
            self.use_gpu = use_gpu
            self.nprobe = nprobe
            self.batch_size = batch_size
            self.n_workers = n_workers or min(os.cpu_count(), 8)
            self.labels_ = None
        
    def _build_index(self, embeddings: np.ndarray):
        """构建FAISS索引"""
        with timer_context(self.timer, "构建FAISS索引"):
            log_step("开始构建FAISS索引", self.timer)
            d = embeddings.shape[1]
            n = embeddings.shape[0]
            
            # 使用推荐参数
            params = get_recommended_params(n, d)
            nlist = params['nlist']
            
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            
            with timer_context(self.timer, "训练索引"):
                log_step("训练FAISS索引", self.timer)
                index.train(embeddings)
                index.nprobe = self.nprobe
            
            # 批量添加向量
            with timer_context(self.timer, "添加向量"):
                log_step("添加向量到索引", self.timer)
                for i in tqdm(range(0, n, self.batch_size), desc="添加向量"):
                    batch = embeddings[i:i+self.batch_size]
                    index.add(batch)
            
            return index

    def _process_small_cluster(self, label, embeddings, labels, min_samples, index, k):
        """处理单个小簇"""
        if np.sum(labels == label) < min_samples:
            mask = labels == label
            if not np.any(mask):
                return None
            
            cluster_samples = embeddings[mask]
            mean_vector = np.mean(cluster_samples, axis=0, keepdims=True)
            D, I = index.search(mean_vector, k)
            
            for idx in I[0]:
                target_label = labels[idx]
                if target_label != label and np.sum(labels == target_label) >= min_samples:
                    return (mask, target_label)
        return None

    def fit(self, embeddings: np.ndarray):
        with timer_context(self.timer, "总耗时"):
            log_step("开始聚类", self.timer)
            n_samples = len(embeddings)
            
            # L2归一化
            with timer_context(self.timer, "L2归一化"):
                log_step("执行L2归一化", self.timer)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            
            # 构建索引
            index = self._build_index(embeddings)
            
            # 初始化标签
            with timer_context(self.timer, "初始化标签"):
                log_step("初始化标签", self.timer)
                self.labels_ = np.arange(n_samples)
            
            # 批量计算K近邻
            with timer_context(self.timer, "K近邻搜索"):
                log_step("开始计算K近邻", self.timer)
                k = min(100, n_samples)
                D, I = [], []
                
                for i in tqdm(range(0, n_samples, self.batch_size), desc="K近邻搜索"):
                    batch = embeddings[i:i+self.batch_size]
                    D_batch, I_batch = index.search(batch, k)
                    D.append(D_batch)
                    I.append(I_batch)
                
                D = np.vstack(D)
                I = np.vstack(I)
            
            # 构建相似度图
            with timer_context(self.timer, "构建相似度图"):
                log_step("开始构建相似度图", self.timer)
                similar_pairs = []
                # 使用向量化操作
                mask = D >= self.similarity_threshold
                rows, cols = np.where(mask)
                for row, col in zip(rows, cols):
                    if row < I[row, col]:  # 避免重复对
                        similar_pairs.append((D[row, col], row, I[row, col]))
                
                log_step(f"相似度图构建完成，共{len(similar_pairs)}对相似向量", self.timer)
            
            # 合并簇
            with timer_context(self.timer, "合并簇"):
                log_step("开始合并簇", self.timer)
                similar_pairs.sort(reverse=True)
                for sim, i, j in tqdm(similar_pairs, desc="合并簇"):
                    if self.labels_[i] != self.labels_[j]:
                        cluster1 = self.labels_[i]
                        cluster2 = self.labels_[j]
                        
                        size1 = np.sum(self.labels_ == cluster1)
                        size2 = np.sum(self.labels_ == cluster2)
                        
                        if size1 >= self.min_samples and size2 >= self.min_samples:
                            continue
                        
                        old_label = max(cluster1, cluster2)
                        new_label = min(cluster1, cluster2)
                        self.labels_[self.labels_ == old_label] = new_label

            # 检测噪声点
            with timer_context(self.timer, "检测噪声点"):
                log_step("开始检测噪声点", self.timer)
                
                # 获取每个点的邻居数量
                neighbor_counts = np.zeros(n_samples)
                for i in range(n_samples):
                    # 计算与阈值以上的邻居数量
                    neighbor_counts[i] = np.sum(D[i] >= self.similarity_threshold)
                
                # 标记噪声点
                noise_mask = neighbor_counts < self.min_samples
                
                # 获取所有簇的大小
                unique_labels, cluster_sizes = np.unique(self.labels_, return_counts=True)
                small_clusters = unique_labels[cluster_sizes < self.min_samples]
                
                # 将小簇中的点也标记为噪声
                for label in small_clusters:
                    noise_mask |= (self.labels_ == label)
                
                # 将噪声点的标签设为-1
                self.labels_[noise_mask] = -1
                
                log_step(f"检测到{np.sum(noise_mask)}个噪声点", self.timer)
            
            # 重新标记簇号
            with timer_context(self.timer, "重新标记簇号"):
                log_step("重新标记簇号", self.timer)
                unique_labels = np.unique(self.labels_)
                # 排除噪声标签-1
                unique_labels = unique_labels[unique_labels != -1]
                label_map = {old: new for new, old in enumerate(unique_labels)}
                # 保持噪声点的标签为-1
                self.labels_ = np.array([label_map.get(x, -1) for x in self.labels_])
            
            # 清理内存
            del D, I
            gc.collect()
            
            log_step(f"聚类完成，共{len(np.unique(self.labels_[self.labels_ != -1]))}个簇，{np.sum(self.labels_ == -1)}个噪声点", self.timer)
            
            # 输出性能统计
            self.timer.summary()
            
            return self

def fast_cluster_embeddings(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.8,
    min_samples: int = 5,
    use_gpu: bool = True,
    nprobe: int = None,
    batch_size: int = None,
    n_workers: int = None
) -> np.ndarray:
    """快速聚类接口"""
    # 获取推荐参数
    params = get_recommended_params(len(embeddings), embeddings.shape[1])
    
    # 使用推荐参数或用户指定参数
    nprobe = nprobe or params['nprobe']
    batch_size = batch_size or params['batch_size']
    n_workers = n_workers or params['n_workers']
    
    logging.info(f"Recommended Nprobe: {nprobe}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Workers: {n_workers}")
    
    clusterer = FastThresholdClustering(
        similarity_threshold=similarity_threshold,
        min_samples=min_samples,
        use_gpu=use_gpu,
        nprobe=nprobe,
        batch_size=batch_size,
        n_workers=n_workers
    )
    return clusterer.fit(embeddings).labels_
