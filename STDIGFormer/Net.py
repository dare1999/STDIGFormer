
import torch

import torch.nn as nn
import copy
import torch.nn.functional as F
import math


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'PN':
            col_mean = x.mean(dim=0)
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        return x


class GraphNorm(nn.Module):
    """
    Graph-wise normalization: normalize node features across the graph (node
    dimension). Learnable scale γ and shift β maintain feature diversity.
    """
    def __init__(self, d_model, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model))   # γ
        self.b = nn.Parameter(torch.zeros(d_model))  # β
        self.eps = eps

    def forward(self, x):          # x: [N, d_model]
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.g * x_norm + self.b


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with
      • Top-k sparse multi-head attention
      • GraphNorm (replaces PairNorm)
      • Optional gated memory (LSTM) per node
      • Cross-layer initial residual handled in GraphNet wrapper
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hop_k: int = 1,
        dropout: float = 0.1,
        top_k: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.top_k = top_k                     # Top-k similar nodes per head

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # Edge bias projection (num_heads channels)
        self.W_e = nn.Linear(d_model, num_heads)
        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)

        # Normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.graphnorm = GraphNorm(d_model)

        # Position-wise FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Gated memory (optional)
        self.memory_lstm = nn.LSTMCell(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h, e, adjm_bin, memory_state=None):
        """
        h         : [N, d_model] node features
        e         : [N, N, d_model] edge embeddings
        adjm_bin  : [N, N] binary adjacency (not used for sparsity here)
        memory_state : tuple(h_mem, c_mem) from previous step
        """
        N = h.size(0)
        residual = h

        # --- Multi-head QKV --------------------------------------------------
        q = self.W_q(h).view(N, self.num_heads, self.head_dim)  # [N,H,Dh]
        k = self.W_k(h).view(N, self.num_heads, self.head_dim)
        v = self.W_v(h).view(N, self.num_heads, self.head_dim)

        # Scaled dot-product
        scores = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(self.head_dim)

        # Add edge biases
        scores = scores + self.W_e(e)                     # [N,N,H]

        # --- Top-k sparse mask ----------------------------------------------
        mask_topk = torch.zeros_like(scores, dtype=torch.bool)
        for h_idx in range(self.num_heads):
            _, topk_idx = torch.topk(scores[:, :, h_idx], self.top_k, dim=1)
            mask_topk[:, :, h_idx].scatter_(1, topk_idx, True)

        scores = scores.masked_fill(~mask_topk, float("-inf"))

        # Attention weights
        attn = F.softmax(scores, dim=1)
        attn = self.dropout(attn)

        # Aggregate
        h_attn = torch.einsum("ijh,jhd->ihd", attn, v).contiguous().view(N, self.d_model)
        h_attn = self.out_linear(h_attn)

        # Add & Norm-1
        h = self.norm1(residual + self.dropout(h_attn))

        # Optional gated memory
        if memory_state is not None:
            h_mem, c_mem = memory_state
            h_next, c_next = self.memory_lstm(h, (h_mem, c_mem))
            h = h + h_mem + h_next
            memory_state = (h_next, c_next)
        else:
            h_next, c_next = self.memory_lstm(
                h, (torch.zeros_like(h), torch.zeros_like(h))
            )
            h = h + h_next
            memory_state = (h_next, c_next)

        # GraphNorm
        h = self.graphnorm(h)

        # FFN + Add & Norm-2
        h_ffn = self.ffn(h)
        h = self.norm2(h + self.dropout(h_ffn))

        return h, memory_state


class GraphNet(nn.Module):
    """
    Stacks multiple GraphTransformerLayer blocks with
    • initial embedding
    • cross-layer initial residual (h0 skip)
    """
    def __init__(
        self,
        inputdims: int,
        headnums: int,
        projectdims: int,
        layernums: int,
        hop_k: int = 2,
        dropout: float = 0.1,
        top_k: int = 5,
    ):
        super().__init__()
        self.d_model = headnums * projectdims
        self.layernums = layernums

        # Node / edge embeddings
        self.embedding = nn.Linear(inputdims, self.d_model)
        self.edge_embedding = nn.Linear(1, self.d_model)

        # Transformer stack
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    self.d_model, headnums, hop_k, dropout, top_k
                )
                for _ in range(layernums)
            ]
        )

        self.output_layer = nn.Linear(self.d_model, self.d_model)

    def forward(self, nodefeature, adjm, returnlinearweights=False):
        """
        nodefeature : [N, F]
        adjm        : [N, N] adjacency (float / int)
        """
        # Initial embed
        h = self.embedding(nodefeature)
        h0 = h.clone()                          # cross-layer skip

        # Edge embed (binary)
        adjm_bin = (adjm > 0).float()
        e = self.edge_embedding(adjm_bin.unsqueeze(-1))

        # Memory state list per layer
        memory_states = [None] * self.layernums

        # Stack layers
        for i, layer in enumerate(self.layers):
            h, memory_states[i] = layer(h, e, adjm_bin, memory_states[i])
            # Cross-layer initial residual
            h = h + h0

        # Output projection
        h = self.output_layer(h)
        if returnlinearweights:
            return h, [self.output_layer.weight]
        else:
            return h

def clones(module, N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)


    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]


        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Applies a sublayer to x and returns the result with residual connection and layer normalization."
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Pass the input through the self-attention and feed-forward layers with residual connections and layer normalization."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        return self.sublayer[1](x, self.feed_forward)

    def forward_(self,x1,x2,x3,mask):

        x = self.sublayer[0](x3, lambda x: self.self_attn(x1, x2, x3, mask))
        "Pass the input through the self-attention and feed-forward layers with residual connections and layer normalization."
        return self.sublayer[1](x, self.feed_forward)
class Transformer(nn.Module):
    def __init__(self, inputdims, headnums, projectdims, layernums):
        super().__init__()
        self.firstlinear = nn.Linear(inputdims, headnums * projectdims)
        self.encoderlayers = []

        for i in range(layernums):
            mulatten = MultiHeadedAttention(headnums, projectdims * headnums)
            self.encoderlayers.append(EncoderLayer(headnums * projectdims, mulatten, nn.Linear(headnums * projectdims, headnums * projectdims), 0.5))
            self.add_module('encoderlayer' + str(i), self.encoderlayers[-1])

        self.headnums = headnums
    def forward(self, x,pos, ϕ=0.5, γ=0.5, μ=0.6, ξ=0.3, τ=0.2):
        x = self.firstlinear(x)
        pos_encode = self.get_pos_encode(x.shape[0], x.shape[1], x.shape[2])
        x = x + pos_encode

        # 动态窗口稀疏注意力
        mask = self.dynamic_window_mask(pos, base_ratio=0.1,
                                        phi=ϕ, gamma=γ, mu=μ, xi=ξ, tau=τ)
        mask = mask.to(x.device)

        for i in range(len(self.encoderlayers)):
            x = self.encoderlayers[i](x, mask)
        return x

    def dynamic_window_mask(self, pos, base_ratio, phi, gamma, mu, xi, tau):

        B, T, _ = pos.size()
        device = pos.device

        delta = pos[:, 1:, :] - pos[:, :-1, :]  # [B, T-1, 2]
        speed = torch.norm(delta, dim=-1)  # [B, T-1]
        v = torch.zeros(B, T, device=device)
        v[:, 1:] = speed
        dv = torch.zeros_like(v)
        dv[:, 1:] = torch.abs(v[:, 1:] - v[:, :-1])  # [B, T]

        angles = torch.zeros(B, T, device=device)
        angles[:, 1:] = torch.atan2(delta[:, :, 1], delta[:, :, 0])  # [B, T]
        dtheta = torch.zeros_like(angles)
        dtheta[:, 1:] = torch.abs(angles[:, 1:] - angles[:, :-1])  # [B, T]

        Phi = phi * dv + gamma * dtheta  # [B, T]
        EF = torch.exp(- (Phi - mu) ** 2 / (2 * xi ** 2))  # [B, T]
        BS = int(base_ratio * T)
        w = (BS + EF).long()  # [B, T]

        mask = torch.zeros(B, T, T, dtype=torch.bool, device=device)
        for b in range(B):
            for t in range(T):
                wt = w[b, t].item()
                if wt >= tau:
                    start = max(0, t - wt // 2)
                    end = min(T, t + wt // 2 + 1)
                    mask[b, t, start:end] = True
                else:
                    mask[b, t, t] = True
        return mask

    def get_pos_encode(self, batchsize, length, dmodel):
        pos_encode = torch.zeros(length, dmodel).float()
        pos_encode[:, 0::2] = torch.sin(torch.arange(length).float().unsqueeze(1).mm(1000 ** (-1.0 * torch.arange(0, dmodel, 2).unsqueeze(0) / dmodel)))
        pos_encode[:, 1::2] = torch.cos(torch.arange(length).float().unsqueeze(1).mm(1000 ** (-1.0 * torch.arange(1, dmodel, 2).unsqueeze(0) / dmodel)))
        pos_encode.unsqueeze(0).repeat(batchsize, 1, 1)
        return pos_encode
class MyNet(nn.Module):

    def __init__(self,peoplenums,embedding_dim,inputdims,gatheadnums,gatprojectdims,gatlayernums,adjnums,transformerheadnums,transformerprojectdims,transformerlayernums,outputdims):

        super(MyNet,self).__init__()
        self.linear_map_g = nn.Linear(adjnums * gatheadnums * gatprojectdims,
                                      transformerheadnums * transformerprojectdims)
        self.linear_map_t = nn.Linear(transformerheadnums * transformerprojectdims,
                                      transformerheadnums * transformerprojectdims)
        self.layer_norm = nn.LayerNorm(transformerheadnums * transformerprojectdims)

        self.graphnets=[]
        self.depthwise_conv1 = nn.Conv2d(in_channels=transformerheadnums * transformerprojectdims,
                                         out_channels=transformerheadnums * transformerprojectdims,
                                         kernel_size=(3, 3), padding=1,
                                         groups=transformerheadnums * transformerprojectdims)
        self.pointwise_conv1 = nn.Conv2d(in_channels=transformerheadnums * transformerprojectdims,
                                         out_channels=64, kernel_size=1)

        self.depthwise_conv2 = nn.Conv2d(in_channels=transformerheadnums * transformerprojectdims,
                                         out_channels=transformerheadnums * transformerprojectdims,
                                         kernel_size=(5, 5), padding=2,
                                         groups=transformerheadnums * transformerprojectdims)
        self.pointwise_conv2 = nn.Conv2d(in_channels=transformerheadnums * transformerprojectdims,
                                         out_channels=64, kernel_size=1)

        self.depthwise_conv3 = nn.Conv2d(in_channels=transformerheadnums * transformerprojectdims,
                                         out_channels=transformerheadnums * transformerprojectdims,
                                         kernel_size=(7, 7), padding=3,
                                         groups=transformerheadnums * transformerprojectdims)
        self.pointwise_conv3 = nn.Conv2d(in_channels=transformerheadnums * transformerprojectdims,
                                         out_channels=64, kernel_size=1)

        # SE Block
        self.se_block = SEBlock(channel=64 * 3, reduction=16)

        # BatchNorm
        self.batch_norm = nn.BatchNorm2d(64 * 3)

        self.time_conv = nn.Conv2d(64 * 3, outputdims, kernel_size=(1, 1))

        for i in range(adjnums):
            self.graphnets.append(GraphNet(inputdims,gatheadnums,gatprojectdims,gatlayernums))
            self.add_module('graphnet'+str(i),self.graphnets[-1])

        self.gatheadnums=gatheadnums
        self.gatprojectdims=gatprojectdims

        self.transformer=Transformer(gatheadnums*gatprojectdims*adjnums,transformerheadnums,transformerprojectdims,transformerlayernums)

        self.linear1=nn.Linear(transformerheadnums*transformerprojectdims,120)

        self.linear2=nn.Linear(120,outputdims)

        self.emb=nn.Embedding(peoplenums, embedding_dim)
        self.refining = nn.Sequential(
            nn.Conv1d(outputdims, 12, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(12, 6, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(6, 6, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(6, outputdims, kernel_size=1)
        )
        self.cnn = CNN(1, 12)
        self.linear1 = nn.Linear(outputdims, 120)
        self.linear2 = nn.Linear(120, outputdims)

    def forward(self,datax,adjms,returnlinearweights=False):

        embfeature=torch.arange(datax.shape[0])
        embfeature=self.emb(embfeature)

        graphnetgoutput=[]
        graphnetlinearweights=[]
        for i in range(len(adjms)):
            onetimefeature=[]
            graphnetlinearweights_=[]
            for j in range(datax.shape[1]):
                if returnlinearweights:
                    nodefeature,graphnetlinearweights_=self.graphnets[i](torch.cat([datax[:,j,:],embfeature],-1),adjms[i][:,:,j],returnlinearweights=True)
                    onetimefeature.append(nodefeature)
                else:
                    nodefeature=self.graphnets[i](torch.cat([datax[:,j,:],embfeature],-1),adjms[i][:,:,j])
                    onetimefeature.append(nodefeature)

            graphnetlinearweights.extend(graphnetlinearweights_)

            oneadjfeature=torch.cat([_.unsqueeze(1) for _ in onetimefeature],1)

            graphnetgoutput.append(oneadjfeature)

        graphnetgoutput=torch.cat(graphnetgoutput,-1)

        '''
        o=nn.Tanh()(self.linear3(graphnetgoutput[:,-1,:]))
        o=self.linear4(o)
        if returnlinearweights:
            return o,graphnetlinearweights
        else:
            return o
        '''
        o1=self.transformer(graphnetgoutput,datax)

        g = self.linear_map_g(graphnetgoutput)
        def scaled_dot_product_attention(query, key, value, mask=None):
            d_k = query.size(-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim=-1)
            return torch.matmul(p_attn, value), p_attn

        context_t2g, _ = scaled_dot_product_attention(o1, g, g)
        context_g2t, _ = scaled_dot_product_attention(g, o1, o1)
        o1 = self.layer_norm(o1 + context_t2g + context_g2t)

        o1 = o1.permute(0, 2, 1).unsqueeze(3)
        out1 = self.depthwise_conv1(o1)
        out1 = F.relu(out1)
        out1 = self.pointwise_conv1(out1)

        out2 = self.depthwise_conv2(o1)
        out2 = F.relu(out2)
        out2 = self.pointwise_conv2(out2)

        out3 = self.depthwise_conv3(o1)
        out3 = F.relu(out3)
        out3 = self.pointwise_conv3(out3)

        concat = torch.cat([out1, out2, out3], dim=1)

        # SE Block
        concat = self.se_block(concat)

        # BatchNorm
        concat = self.batch_norm(concat)

        o2 = self.time_conv(concat)

        o2 = o2.squeeze(3).permute(0, 2, 1)  # [batch_size, history_times, outputdims]

        o = o2.permute(0, 2, 1)  # [batch_size, outputdims, history_times]
        o = self.refining(o)
        o = o.permute(0, 2, 1)  # [batch_size, history_times, outputdims]
        o = o[:, -1, :]  # [batch_size, outputdims]

        o = nn.Sigmoid()(self.linear1(o))
        o = nn.Sigmoid()(self.linear2(o))
        xs = max(o[:, 0::2]) * o[:, 0::2] - min(o[:, 0::2])
        ys = max(o[:, 1::2]) * o[:, 1::2] - min(o[:, 1::2])

        o = []
        for i in range(xs.shape[1]):
            o.append(xs[:, i:i + 1])
            o.append(ys[:, i:i + 1])
        o = torch.cat(o, -1)

        if returnlinearweights:
            return o, graphnetlinearweights
        else:
            return o

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def compute_projectdiffloss(linearweights):

    projectdiffloss=[]
    for linearweight in linearweights:

        L2m=torch.norm(linearweight,2,-1,keepdim=True)
        projectdiffloss.append(torch.mean(linearweight.mm(linearweight.transpose(0,1))/L2m.mm(L2m.transpose(0,1))))

    return sum(projectdiffloss)/len(projectdiffloss)


class MLP(nn.Module):
    def __init__(self, in_size, out_size=None, normalization=False, act_name='prelu'):
        super(MLP, self).__init__()
        if out_size is None:
            out_size = in_size
        self.linear = nn.Linear(in_size, out_size)
        self.ln = LayerNorm(out_size) if normalization else nn.Sequential()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.ln(x)
        x = self.activation(x)
        return x


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)

        return out


class CNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(256 * 8 * 45, 512)
        self.fc2 = nn.Linear(512, output_channels * 8)
        self.output_channels = output_channels

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 8, 45)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 8, self.output_channels)  # Reshape to (batch_size, 8, output_channels)
        return x






