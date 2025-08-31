import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import MoE_multi_trm

class MISSRec(MoE_multi_trm):
    # 计算损失主入口
    def calculate_loss(self, interaction):
        # 预训练阶段损失
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)
        # 微调阶段损失
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, interest_orthogonal_regularization = self._compute_seq_embeddings(item_seq, item_seq_len)
        # 如果有多模态（文本+图片），做加权融合
        if 'text' in self.modal_type and 'img' in self.modal_type:
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            logits = self._compute_dynamic_fused_logits(seq_output, test_text_emb, test_img_emb) / self.temperature
        else:
            # 单模态直接算logits
            test_item_emb = self._compute_test_item_embeddings()
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature

        pos_items = interaction[self.POS_ITEM_ID]
        # 总损失=主损失+正则项
        loss = self.loss_fct(logits, pos_items) + self.gamma * interest_orthogonal_regularization
        return loss

    # 预训练阶段损失
    def pretrain(self, interaction):
        # adapter
        img_emb = self.img_adaptor(interaction['img_emb_list'])
        test_emb = self.text_adaptor(interaction['text_emb_list'])
        # 计算序列输出和正则项
        seq_output, interest_orthogonal_regularization = self._compute_seq_embeddings_pretrain(
            item_seq=interaction[self.ITEM_SEQ],
            item_seq_len=interaction[self.ITEM_SEQ_LEN],
            text_emb=self.text_adaptor(interaction['text_emb_list']),
            img_emb=self.img_adaptor(interaction['img_emb_list']),
            text_emb_empty_mask=interaction['text_emb_empty_mask_list'],
            img_emb_empty_mask=interaction['img_emb_empty_mask_list'],
            unique_interest_seq=interaction['unique_interest_list'],
            unique_interest_emb_list=interaction['unique_interest_emb_list'],
            unique_interest_len=interaction['unique_interest_len']
        )
        batch_size = seq_output.shape[0]
        device = seq_output.device
        batch_labels = torch.arange(batch_size, device=device, dtype=torch.long)

        # 序列-物品对比损失
        loss_seq_item = self.seq_item_contrastive_task(seq_output, interaction, batch_labels)
        # 序列-序列对比损失
        loss_seq_seq, interest_orthogonal_regularization_aug = self.seq_seq_contrastive_task(
            seq_output, interaction, img_emb, batch_labels)
        # 总损失=序列-物品 + λ*序列-序列 + γ*正则项
        loss = loss_seq_item + self.lam * loss_seq_seq + self.gamma * (
                    interest_orthogonal_regularization + interest_orthogonal_regularization_aug)
        return loss

    # 全排序预测（评估用）
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self._compute_seq_embeddings(item_seq, item_seq_len)
        # 多模态融合
        if 'text' in self.modal_type and 'img' in self.modal_type:
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            scores = self._compute_dynamic_fused_logits(seq_output, test_text_emb, test_img_emb) / self.temperature
        else:
            test_item_emb = self._compute_test_item_embeddings()
            scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        return scores

    # 计算序列embedding（微调/评估用）
    def _compute_seq_embeddings(self, item_seq, item_seq_len):
        # 获取文本模态embedding
        if 'text' in self.modal_type:
            text_emb = self.text_adaptor(self.plm_embedding(item_seq))  # [B, L, D]
            text_emb_empty_mask = self.plm_embedding_empty_mask[item_seq]  # [B, L]
        # 获取图片模态embedding
        if 'img' in self.modal_type:
            img_emb = self.img_adaptor(self.img_embedding(item_seq))  # [B, L, D]
            img_emb_empty_mask = self.img_embedding_empty_mask[item_seq]  # [B, L]

        # 不同融合方式初始化
        item_emb_list = 0 if self.seq_mm_fusion == 'add' else []
        item_modal_empty_mask_list = []
        interest_seq_list = []

        # 文本模态处理
        if 'text' in self.modal_type:
            if self.seq_mm_fusion == 'add':
                item_emb_list = item_emb_list + text_emb
            else:
                item_emb_list.append(text_emb)
            item_modal_empty_mask_list.append(text_emb_empty_mask)
            plm_interest_seq = self.plm_interest_lookup_table[item_seq]  # [B, L]
            interest_seq_list.append(plm_interest_seq)

        # 图片模态处理
        if 'img' in self.modal_type:
            if self.seq_mm_fusion == 'add':
                item_emb_list = item_emb_list + img_emb
            else:
                item_emb_list.append(img_emb)
            item_modal_empty_mask_list.append(img_emb_empty_mask)
            img_interest_seq = self.img_interest_lookup_table[item_seq]  # [B, L]
            interest_seq_list.append(img_interest_seq)

        # 多模态融合
        if self.seq_mm_fusion != 'add':
            item_emb_list = torch.stack(item_emb_list, dim=1)
        item_modal_empty_mask = torch.stack(item_modal_empty_mask_list, dim=1)

        # 拼接兴趣序列
        all_interest_seq = torch.cat(interest_seq_list, dim=-1)
        unique_interest_seq = []
        unique_interest_len = []
        # 对每个样本兴趣去重
        for sample in all_interest_seq:
            unique_interests = sample.unique()
            unique_interest_len.append(len(unique_interests))
            unique_interest_seq.append(unique_interests)
        # pad到同长度
        unique_interest_seq = nn.utils.rnn.pad_sequence(unique_interest_seq, batch_first=True,
                                                        padding_value=0)  # [B, Nq]
        unique_interest_emb_list = self.interest_embeddings[unique_interest_seq]  # [B, Nq, D]
        unique_interest_len = torch.tensor(unique_interest_len, device=unique_interest_seq.device)
        del interest_seq_list

        # 送入主模型forward
        seq_output, interest_orthogonal_regularization = self.forward(
            item_seq=item_seq,
            item_emb=item_emb_list,
            item_modal_empty_mask=item_modal_empty_mask,
            item_seq_len=item_seq_len,
            interest_seq=unique_interest_seq,
            interest_emb=unique_interest_emb_list,
            interest_seq_len=unique_interest_len
        )
        seq_output = F.normalize(seq_output, dim=1)
        return seq_output, interest_orthogonal_regularization

    # 计算序列embedding（预训练用）
    def _compute_seq_embeddings_pretrain(
            self, item_seq, item_seq_len,
            text_emb, img_emb,
            text_emb_empty_mask=None,
            img_emb_empty_mask=None,
            # text_interest_seq=None,
            # img_interest_seq=None
            unique_interest_seq=None,
            unique_interest_emb_list=None,
            unique_interest_len=None
    ):
        item_emb_list = 0 if self.seq_mm_fusion == 'add' else []
        item_modal_empty_mask_list = []

        if 'text' in self.modal_type:
            if self.seq_mm_fusion == 'add':
                item_emb_list = item_emb_list + text_emb
            else:
                item_emb_list.append(text_emb)
            item_modal_empty_mask_list.append(text_emb_empty_mask)

        if 'img' in self.modal_type:
            if self.seq_mm_fusion == 'add':
                item_emb_list = item_emb_list + img_emb
            else:
                item_emb_list.append(img_emb)
            item_modal_empty_mask_list.append(img_emb_empty_mask)

        if self.seq_mm_fusion != 'add':
            item_emb_list = torch.stack(item_emb_list, dim=1)
        item_modal_empty_mask = torch.stack(item_modal_empty_mask_list, dim=1)

        # 送入主模型forward
        seq_output, interest_orthogonal_regularization = self.forward(
            item_seq=item_seq,
            item_emb=item_emb_list,
            item_modal_empty_mask=item_modal_empty_mask,
            item_seq_len=item_seq_len,
            interest_seq=unique_interest_seq,
            interest_emb=unique_interest_emb_list,
            interest_seq_len=unique_interest_len
        )
        seq_output = F.normalize(seq_output, dim=1)
        return seq_output, interest_orthogonal_regularization

    # Transformer主结构
    def forward(self,
                item_seq,
                item_emb,
                item_modal_empty_mask,
                item_seq_len,
                interest_seq=None,
                interest_emb=None,
                interest_seq_len=None):
        enc_input_emb = interest_emb
        src_attn_mask, src_key_padding_mask = self.get_encoder_attention_mask(interest_seq, is_casual=False)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_embedding = self.position_embedding(position_ids)  # [LxD]
        # 位置编码扩展到batch
        position_embedding = position_embedding.unsqueeze(0).expand(item_emb.size(0), -1, -1)
        dec_input_emb = item_emb + position_embedding  # [BxLxD]
        # 微调阶段加id embedding
        if self.train_stage == 'transductive_ft' and self.id_type != 'none':
            item_id_embeddings = self.item_embedding(item_seq)
            if self.seq_mm_fusion != 'add':
                item_id_embeddings = item_id_embeddings.unsqueeze(1)  # [Bx1xLxD]
            dec_input_emb = dec_input_emb + item_id_embeddings
        # 多模态融合展开
        if self.seq_mm_fusion != 'add':
            dec_input_emb = dec_input_emb.view(dec_input_emb.size(0), -1,
                                               dec_input_emb.size(-1))  # [BxMxLxD] => [Bx(M*L)xD]
        dec_input_emb = self.LayerNorm(dec_input_emb)
        dec_input_emb = self.dropout(dec_input_emb)
        tgt_attn_mask, tgt_cross_attn_mask, tgt_key_padding_mask = self.get_decoder_attention_mask(
            item_seq, item_modal_empty_mask, is_casual=False)
        memory_key_padding_mask = src_key_padding_mask
        
        # 编码器（兴趣序列）
        memory = self.encoder(
            src=enc_input_emb,  # [B, L, D] - 不转置
            mask=src_attn_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        # pool兴趣embedding
        src_key_token_weight = (~src_key_padding_mask).unsqueeze(-1).float().mean(1,
                                                                                  keepdim=True)  # [BxL] => [BxLx1] => [Bx1x1]
        pooled_memory = (memory * src_key_token_weight).sum(1)  # ([BxLxD] * [Bx1x1]) => [BxD]
        # 正交正则项
        interest_orthogonal_regularization = (pooled_memory * pooled_memory).sum() / pooled_memory.shape[
            1]  # [BxD] x [BxD] => [B]
        # MoE解码器
        moe_output, moe_aux_loss = self.moe_decoder(
            dec_input_emb,
            memory,
            tgt_mask=tgt_attn_mask,
            memory_mask=tgt_cross_attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        #print("MoE融合输出"
        #print(moe_output.shape)
        return moe_output, interest_orthogonal_regularization.mean() + moe_aux_loss# [BXD],[]

    # 序列与正样本物品对比损失
    def seq_item_contrastive_task(self, seq_output, interaction, batch_labels):
        if 'text' in self.modal_type:
            pos_text_emb = self.text_adaptor(interaction['pos_text_emb'])
        if 'img' in self.modal_type:
            pos_img_emb = self.img_adaptor(interaction['pos_img_emb'])
        if 'text' in self.modal_type and 'img' in self.modal_type:
            logits = self._compute_dynamic_fused_logits(seq_output, pos_text_emb, pos_img_emb) / self.temperature
        else:
            if 'text' in self.modal_type:
                pos_item_emb = pos_text_emb
            if 'img' in self.modal_type:
                pos_item_emb = pos_img_emb
            pos_item_emb = F.normalize(pos_item_emb, dim=1)
            logits = torch.matmul(seq_output, pos_item_emb.transpose(0, 1)) / self.temperature
        loss = F.cross_entropy(logits, batch_labels)
        return loss

    # 序列与增强序列对比损失
    def seq_seq_contrastive_task(self, seq_output, interaction, img_emb, batch_labels):
        seq_output_aug, interest_orthogonal_regularization_aug = self._compute_seq_embeddings_pretrain(
            item_seq=interaction[self.ITEM_SEQ + '_aug'],
            item_seq_len=interaction[self.ITEM_SEQ_LEN + '_aug'],
            text_emb=self.text_adaptor(interaction['text_emb_list_aug']),
            img_emb=img_emb,
            text_emb_empty_mask=interaction['text_emb_empty_mask_list_aug'],
            img_emb_empty_mask=interaction['img_emb_empty_mask_list'],
            unique_interest_seq=interaction['unique_interest_list_aug'],
            unique_interest_emb_list=interaction['unique_interest_emb_list_aug'],
            unique_interest_len=interaction['unique_interest_len_aug']
        )
        logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        loss = F.cross_entropy(logits, batch_labels)
        return loss, interest_orthogonal_regularization_aug

    # 获取所有测试物品embedding
    def _compute_test_item_embeddings(self):
        test_item_emb = 0
        if 'text' in self.modal_type:
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_item_emb = test_item_emb + test_text_emb
        if 'img' in self.modal_type:
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            test_item_emb = test_item_emb + test_img_emb

        if self.train_stage == 'transductive_ft':
            if self.id_type != 'none':
                test_item_emb = test_item_emb + self.item_embedding.weight

        test_item_emb = F.normalize(test_item_emb, dim=1)
        return test_item_emb

    # 多模态融合logits
    def _compute_dynamic_fused_logits(self, seq_output, text_emb, img_emb):
        text_emb = F.normalize(text_emb, dim=1)
        img_emb = F.normalize(img_emb, dim=1)
        text_logits = torch.matmul(seq_output, text_emb.transpose(0, 1))  # [BxB]
        img_logits = torch.matmul(seq_output, img_emb.transpose(0, 1))  # [BxB]
        modality_logits = torch.stack([text_logits, img_logits], dim=-1)  # [BxBx2]
        # 动态融合方式
        if self.item_mm_fusion in ['dynamic_shared', 'dynamic_instance']:
            agg_logits = (modality_logits * F.softmax(modality_logits * self.fusion_factor.unsqueeze(-1), dim=-1)).sum(
                dim=-1)
        else:  # 静态融合
            agg_logits = modality_logits.mean(dim=-1)
        # id融合
        if self.train_stage == 'transductive_ft':
            if self.id_type != 'none':
                test_id_emb = F.normalize(self.item_embedding.weight, dim=1)
                id_logits = torch.matmul(seq_output, test_id_emb.transpose(0, 1))
                agg_logits = (id_logits + agg_logits * 2) / 3
        return agg_logits

    # 编码器注意力mask
    def get_encoder_attention_mask(self, dec_input_seq=None, is_casual=True):
        key_padding_mask = (dec_input_seq == 0)
        dec_seq_len = dec_input_seq.size(-1)
        attn_mask = torch.triu(torch.full((dec_seq_len, dec_seq_len), float('-inf'), device=dec_input_seq.device),
                               diagonal=1) if is_casual else None
        return attn_mask, key_padding_mask

    # 解码器注意力mask
    def get_decoder_attention_mask(self, enc_input_seq, item_modal_empty_mask, is_casual=True):
        assert enc_input_seq.size(0) == item_modal_empty_mask.size(0)
        assert enc_input_seq.size(-1) == item_modal_empty_mask.size(-1)
        batch_size, num_modality, seq_len = item_modal_empty_mask.shape
        if self.seq_mm_fusion == 'add':
            key_padding_mask = (enc_input_seq == 0)
        else:
            key_padding_mask = torch.logical_or((enc_input_seq == 0).unsqueeze(1), item_modal_empty_mask)
            key_padding_mask = key_padding_mask.flatten(1)
        if is_casual:
            attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=enc_input_seq.device),
                                   diagonal=1)
            if self.seq_mm_fusion != 'add':
                attn_mask = torch.tile(attn_mask, (num_modality, num_modality))
        else:
            attn_mask = None
        cross_attn_mask = None
        return attn_mask, cross_attn_mask, key_padding_mask

    # 构造函数：初始化参数和各种embedding
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.gamma = config['gamma']
        self.modal_type = config['modal_type']
        self.id_type = config['id_type']
        self.seq_mm_fusion = config['seq_mm_fusion']
        assert self.seq_mm_fusion in ['add', 'contextual']
        self.item_mm_fusion = config['item_mm_fusion']
        assert self.item_mm_fusion in ['static', 'dynamic_shared', 'dynamic_instance']

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        # embedding初始化
        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            all_num_embeddings = 0
            if 'text' in self.modal_type:
                self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
                # 空embedding mask
                self.register_buffer('plm_embedding_empty_mask', (~self.plm_embedding.weight.data.sum(-1).bool()))
                all_num_embeddings += (self.plm_embedding.num_embeddings - 1)
                # 物品到兴趣lookup表
                self.register_buffer('plm_interest_lookup_table',
                                     torch.zeros(self.plm_embedding.num_embeddings, dtype=torch.long))
            if 'img' in self.modal_type:
                self.img_embedding = copy.deepcopy(dataset.img_embedding)
                self.register_buffer('img_embedding_empty_mask', (~self.img_embedding.weight.data.sum(-1).bool()))
                all_num_embeddings += (self.img_embedding.num_embeddings - 1)
                self.register_buffer('img_interest_lookup_table',
                                     torch.zeros(self.img_embedding.num_embeddings, dtype=torch.long))

            self.num_interest = config["num_experts"]
            self.register_buffer('interest_embeddings',
                                 torch.zeros(self.num_interest + 1, config['hidden_size'], dtype=torch.float))

        # 多模态融合参数
        if 'text' in self.modal_type and 'img' in self.modal_type:
            if self.item_mm_fusion == 'dynamic_shared':
                self.fusion_factor = nn.Parameter(data=torch.tensor(0, dtype=torch.float))
            elif self.item_mm_fusion == 'dynamic_instance':
                self.fusion_factor = nn.Parameter(data=torch.zeros(self.n_items, dtype=torch.float))

        # 适配器
        if 'text' in self.modal_type:
            self.text_adaptor = nn.Linear(config['plm_size'], config['hidden_size'])

        if 'img' in self.modal_type:
            self.img_adaptor = nn.Linear(config['img_size'], config['hidden_size'])