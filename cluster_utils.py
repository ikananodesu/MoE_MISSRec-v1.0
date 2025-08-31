import torch
import torch.nn.functional as F

def init_clusters(num_clusters, input_size):
    directions = torch.randn(num_clusters, input_size)
    directions = torch.nn.functional.normalize(directions, p=2, dim=1)  # [N, D] on sphere
    radii = torch.rand(num_clusters) ** (1.0 / input_size)  # [N]
    clusters = directions * radii.unsqueeze(1)  # [N, D]
    return clusters


def find_neighbor(all_embedding, centroids):
    """
    all_embedding: [N, D]  # N个item的embedding
    centroids: [K, D]      # K个聚类中心
    返回: [N] 每个item最近的centroid标签（0~K-1）
    """
    # 归一化
    all_embedding = F.normalize(all_embedding, p=2, dim=1)
    centroids = F.normalize(centroids, p=2, dim=1)
    # 计算余弦相似度 [N, K]
    sim = torch.matmul(all_embedding, centroids.t())
    # 取最大相似度的索引作为标签
    labels = sim.argmax(dim=1)
    labels+=1
    return labels  # [N]