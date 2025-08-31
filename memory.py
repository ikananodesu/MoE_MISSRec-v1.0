import torch

def memory_curve(interaction, emb):
    # 假设输入 emb: [batch_size, seq_len, emb_dim]
    batch_size, seq_len, emb_dim = emb.shape
    device = emb.device

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    item_length = interaction['item_length'].unsqueeze(1)
    distance_from_end = item_length - 1 - position_ids
    distance_from_end = distance_from_end.clamp(min=0)
    k = 0.5
    weight = 1 / (torch.exp(k * (1-distance_from_end/item_length)))  # 逻辑曲线
    mask = (position_ids < item_length).float()
    final_weight = weight * mask
    final_weight = final_weight.unsqueeze(-1)  # [batch_size, seq_len, 1]

    weighted_emb = emb * final_weight
    return weighted_emb