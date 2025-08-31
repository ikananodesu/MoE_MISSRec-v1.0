from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

#Decoder结构设计
class Decoder_Expert(nn.Module):
    def __init__(self, embedding_size, decoder_layers, decoder_heads, output_size, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        # decoder层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_size,
            nhead=decoder_heads,
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers
        )
        self.ffn_norm = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size*2, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, tgt, memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Args:
            tgt: [tgt_len, batch, embedding_size]
            memory: [src_len, batch, embedding_size]
            tgt_mask: [tgt_len, tgt_len] or None
            memory_mask: [tgt_len, src_len] or None
            tgt_key_padding_mask: [batch, tgt_len] or None
            memory_key_padding_mask: [batch, src_len] or None
        """
        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return output  # [tgt_len, batch, output_size]

# ----------- Encoder函数 -----------
def Encoder(hidden_size, n_layers, n_heads, dropout):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_size,
        nhead=n_heads,
        dropout=dropout,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    return encoder

class MoE_Decoder(nn.Module):
    def __init__(
        self,
        experts,           # nn.ModuleList of Decoder_Expert
        cluster,           # [num_experts, D] tensor
        k,                 # int, top-k experts to use
        loss_coef=1e-12,
        lambda_thresh=0.02
    ):
        super().__init__()
        self.experts = experts
        self.num_experts = len(experts)
        self.register_buffer('cluster', cluster)
        self.k = k
        self.loss_coef = loss_coef
        self.lambda_thresh = lambda_thresh

    def forward(
            self,
            dec_input_emb,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        ):
        B, L_mem, D = memory.size()
        L_dec = dec_input_emb.size(1)
        cluster = self.cluster
        moe_input = memory[:, -1, :]  # [B, D]
        gates = F.softmax(moe_input @ cluster.t(), dim=1)  # [B, N]
        # top-k filtering
        gates = gates.clone()
        gates[gates < self.lambda_thresh] = 0.0
        topk_values, topk_indices = gates.topk(self.k, dim=1)
        mask = torch.zeros_like(gates)
        mask.scatter_(1, topk_indices, 1.0)
        gates = gates * mask
        gates_sum = gates.sum(dim=1, keepdim=True)
        gates = gates / (gates_sum + 1e-8)
        # dispatch
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_outputs = []
        #MoE分派器，对于top-k，分配对应输入和掩码
        for i in range(self.num_experts):
            idx = dispatcher.expert_indices[i]
            if idx.numel() == 0:
                expert_outputs.append(torch.zeros(0, D, device=memory.device))
            else:
                expert_dec_input = dec_input_emb[idx, :, :]    # [B,L,D]
                expert_memory    = memory[idx, :, :]           # [B,L,D]
                # 转换为 [L,B,D]，因为 Decoder_Expert 默认 batch_first=False
                expert_dec_input = expert_dec_input.transpose(0, 1)  # [L,B,D]
                expert_memory    = expert_memory.transpose(0, 1)     # [L,B,D]
                expert_tgt_key_padding_mask = tgt_key_padding_mask[idx, :] if tgt_key_padding_mask is not None else None
                expert_memory_key_padding_mask = memory_key_padding_mask[idx, :] if memory_key_padding_mask is not None else None
                expert_out = self.experts[i](
                    expert_dec_input,
                    expert_memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=expert_tgt_key_padding_mask,
                    memory_key_padding_mask=expert_memory_key_padding_mask
                )  # [L,B,D]
                expert_outputs.append(expert_out[-1])  # [B,D]
        y = dispatcher.combine(expert_outputs)  # [B, D]
        gate_sums_per_expert = gates.sum(0)  #expert加权融合
        aux_loss = (gate_sums_per_expert.float().var() / (gate_sums_per_expert.float().mean() ** 2 + 1e-10))
        aux_loss = aux_loss * self.loss_coef
        return y, aux_loss
        
class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """
        patch分配器

        gates: [batch_size, num_experts]，每个元素为soft权重或0
        """
        self.gates = gates
        self.num_experts = num_experts
        self.expert_indices = []
        self.expert_gates = []
        for i in range(num_experts):
            idx = (gates[:, i] > 0).nonzero(as_tuple=False).squeeze(-1)
            self.expert_indices.append(idx)
            self.expert_gates.append(gates[idx, i])

    def dispatch(self, inp):
        """
        inp: [batch_size, input_dim]
        返回每个专家的输入 [expert_batch_size, input_dim]
        """
        return [inp[idx] for idx in self.expert_indices]

    def expert_gates_for_dispatch(self):
        """每个专家分到的样本的权重"""
        return self.expert_gates

    def combine(self, expert_outputs):
        """
        expert_outputs: List，每个元素 [expert_batch_size, output_dim]
        返回 [batch_size, output_dim]
        对于每个样本，收集其被分配到的专家的输出，按权重加权池化
        """
        batch_size = self.gates.size(0)
        # 兼容空专家
        output_dim = 0
        for eo in expert_outputs:
            if eo.size(0) > 0:
                output_dim = eo.size(1)
                break
        final_output = torch.zeros(batch_size, output_dim, device=expert_outputs[0].device)
        for i in range(self.num_experts):
            idx = self.expert_indices[i]
            if idx.numel() == 0:
                continue
            gates = self.expert_gates[i].unsqueeze(1)  # [expert_batch_size, 1]
            out = expert_outputs[i] * gates  # 权重加权
            final_output.index_add_(0, idx, out)
        return final_output

# ----------- MoE主模块 -----------
class MoE_multi_trm(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # 参数加载
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.layer_norm_eps = config['layer_norm_eps']
        self.k = config['k']
        self.num_experts = config['num_experts']
        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']

        # embedding
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # cluster
        cluster = self.init_clusters(self.num_experts, self.hidden_size)
        self.register_buffer('cluster', cluster)

        # encoder
        self.encoder = Encoder(
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.hidden_dropout_prob
        )

        # experts
        self.experts = nn.ModuleList([
            Decoder_Expert(self.hidden_size, self.n_layers, self.n_heads, self.hidden_size)
            for _ in range(self.num_experts)
        ])

        self.moe_decoder = MoE_Decoder(self.experts, self.cluster, self.k)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # loss
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError()
        self.apply(self._init_weights)

    def init_clusters(self, num_clusters, input_size):
        clusters = torch.randn(num_clusters, input_size)
        clusters = F.normalize(clusters, p=2, dim=1)
        return clusters

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)