
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

torch.set_default_tensor_type(torch.DoubleTensor)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param q: Queries, with dimension [B, L_q, D_q]
        :param k: Keys, with dimension [B, L_k, D_k]
        :param v: Values, with dimension [B, L_v, D_v], generally is k
        :param scale: Scaling factor, a float scalar
        :param attn_mask: Masking, with dimension [B, L_q, L_k]
        :return context: Context
        :return attention: Attention
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = ScaledDotProductAttention(attn_drop)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(proj_drop)
        # layer norm after multi-head attention
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # residual connection
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2) \
            .reshape(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2) \
            .reshape(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2) \
            .reshape(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = dim_per_head ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, num_heads, -1, dim_per_head).transpose(1, 2) \
            .reshape(batch_size, -1, num_heads * dim_per_head)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim, ffn_dim, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.act = nn.GELU()
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = self.w2(self.act(self.w1(x)))
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads, ffn_dim, attn_drop=0.0, proj_drop=0.0, feed_drop=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, attn_drop, proj_drop)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, feed_drop)

    def forward(self, input_k, input_v, input_q, attn_mask=None):
        # self attention
        context, attention = self.attention(input_k, input_v, input_q, attn_mask)
        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class EmbeddingModule(nn.Module):
    """ Embedding module of EEG features """
    def __init__(self, in_chans=17, out_chans=80, patch_embed_bias=True):
        super(EmbeddingModule, self).__init__()
        self.proj = nn.Conv1d(in_channels=in_chans, out_channels=out_chans, kernel_size=1, bias=patch_embed_bias)
        self.BN = torch.nn.BatchNorm1d(out_chans)

    def forward(self, x):
        # x dimension: [batch_size, channels, frequency_band]
        out = F.relu(self.BN(self.proj(x)))
        out = out.transpose(2, 1)
        return out


class OuterProductEmbeddingModule(nn.Module):
    """ Outer Product Embedding Module for EOG features """
    def __init__(self, in_length=36):
        super(OuterProductEmbeddingModule, self).__init__()

        self.proj1 = nn.Linear(in_features=in_length, out_features=in_length)
        self.proj2 = nn.Linear(in_features=in_length, out_features=8)
        self.BN1 = torch.nn.BatchNorm1d(in_length)
        self.BN2 = torch.nn.BatchNorm1d(8)

    def forward(self, x):
        # x dimension: [batch_size, num_of_EOG_features]
        x = self.BN1(x)

        _x = torch.sigmoid(self.proj2(x)) + 1
        x = self.proj1(x)
        out = torch.bmm(x[:, :, np.newaxis], _x[:, np.newaxis, :])

        out = self.BN2(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        return out


class FeatureExtractor(nn.Module):
    """ EOG Feature Extractor """
    def __init__(self, out_chans=80):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv1d(8, 16, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, out_chans, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm1d(out_chans)

    def forward(self, x):
        out = x.transpose(2, 1)

        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        out = out.transpose(2, 1)
        return out


class Regression(nn.Module):
    """ Regression layer """
    def __init__(self, hidden_size):
        super(Regression, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.predict = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x dimension: [batch_size, N, C]
        out = self.norm(x)
        out = self.avgpool(out.transpose(1, 2))
        out = torch.flatten(out, 1)
        out = (torch.tanh(self.predict(out)) + 1) / 2
        return out


class VigilanceNet(nn.Module):
    def __init__(
            self,
            hidden_size=80,
            num_heads=4,
            ffn_dim=320,
            attn_drop=0.0,
            proj_drop=0.0,
            feed_drop=0.0,
    ):
        super(VigilanceNet, self).__init__()

        eeg_chan, freq_bands = 17, 5
        eog_feature = 36

        self.feature_embed1 = EmbeddingModule(in_chans=eeg_chan, out_chans=hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, freq_bands, hidden_size))
        trunc_normal_(self.pos_embed, std=0.02)

        self.feature_embed2 = OuterProductEmbeddingModule(in_length=eog_feature)
        self.eog_encoder = FeatureExtractor(out_chans=hidden_size)

        self.encoder1 = EncoderLayer(hidden_size, num_heads, ffn_dim, attn_drop, proj_drop, feed_drop)
        self.encoder2 = EncoderLayer(hidden_size, num_heads, ffn_dim, attn_drop, proj_drop, feed_drop)

        self.predict1 = Regression(hidden_size)
        self.predict2 = Regression(hidden_size)
        self.predict = Regression(hidden_size)

        self.apply(_init_weights)

    def forward(self, x1, x2):
        # x1 is EEG feature, with dimension [batch_size, channels, freq_bands]
        # x2 is EOG feature, with dimension [batch_size, num_of EOG_features]

        x1 = self.feature_embed1(x1)
        x1 += self.pos_embed
        x1, _ = self.encoder1(x1, x1, x1)

        x2 = self.feature_embed2(x2)
        x2 = self.eog_encoder(x2)

        out1 = self.predict1(x1)
        out2 = self.predict2(x2)

        out, _ = self.encoder2(x2, x2, x1)
        out = self.predict(out)

        return out, out1, out2
