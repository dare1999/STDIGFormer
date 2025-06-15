<!-- Google Tag Manager -->
<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-TL5H773C');</script>
<!-- End Google Tag Manager -->
# STDIGFormer
Framework of STDIGFormer mainly consists of graph learning network, Temporal dynamic window sparse attention (DWSA) Transformer, co-attention and multi scale spatial-temporal feature fusion module.
本仓库实现了论文 **“STDIGFormer: A Spatial-Temporal Dynamic Interaction Graph based Transformer Architecture for Pedestrian Trajectory Prediction”** 中提出的行人轨迹预测模型。

![image](https://github.com/user-attachments/assets/1efc270f-4ab8-4e4c-b03b-eb5ab4c612b6)
Framework of STDIGFormer

## 1. 模型概述

STDIGFormer 旨在解决行人轨迹预测中多类型交互（行人间的避让/跟随/组行为）与行人—环境语义交互难以有效融合的问题，同时兼顾 Transformer 全局注意力的高计算复杂度瓶颈。整体框架如图所示，主要包含以下流程：

1. **构建多类型时空交互图**  
   - 空间邻域图（Spatial Neighborhood Graph）  
   - 跨时段组级依赖图（Intertemporal Group Dependency Graph）  
   - 环境相似度图（Environmental Similarity Graph）  
2. **Graph Transformer**  
   - 利用节点与边特征，结合顶点-边融合和**Top-k Sparse Attention**，并加入**门控记忆（Gated Memory）**，生成空间结构增强表示。  
3. **Temporal Transformer**  
   - 基于输入的时空图特征序列，采用**动态窗口稀疏注意力（Dynamic Window Sparse Attention）**机制与**Cross-Layer Interaction**，提取关键的时间步长依赖。  
4. **多尺度特征融合模块（STFM）**  
   - 基于**Co-Attention**实现时空特征的双向加权；  
   - 插入多层不同卷积核尺寸的**Depthwise Separable Convolution**提取多粒度特征；  
   - 通过**Squeeze-and-Excitation**模块重新校准通道权重。  
5. **解码与多任务微调**  
   - 过渡层中加入时序卷积（Temporal Causal Convolution）与残差连接，减弱噪声；  
   - 最终通过两层 MLP（含残差块）生成未来轨迹坐标偏移。  
6. **知识蒸馏**  
   - 采用“教师模型（深层 Graph+Temporal Transformer）→ 学生模型（浅层 Graph+Temporal Transformer）”的蒸馏策略，实现模型轻量化。

  ## 2. 实验及超参配置

### 2.1 数据输入与预处理

- **数据集**：ETH、UCY（包括 five scenes：ETH、HOTEL、UNIV、ZARA1、ZARA2），共 1536 条行人轨迹。  
- **序列长度**：历史 8 帧（3.2 s）→ 未来预测 12 帧（4.8 s）。  
- **坐标归一化**：将世界坐标（以米或像素为单位）归一化到 $[-1,1]$ 区间；也可直接输入原始坐标。  
- **批处理大小**：8 梯度累积批大小可按需求调整。  

### 3.2 网络结构配置

以下均为论文中“最佳”配置，供复现时参考。

| 模块                     | 教师模型配置              | 学生模型配置              |
|:-------------------------|:-------------------------|:-------------------------|
| **Graph Transformer**    | 4 层 × 3 头，节点维度 16，边维度 16<br>Dropout 0.1 | 2 层 × 3 头，节点维度 16，边维度 16<br>Dropout 0.1 |
| **Temporal Transformer** | 6 层 × 6 头，输入/输出维度 12<br>Cross-Layer Interaction | 3 层 × 6 头，输入/输出维度 12<br>Cross-Layer Interaction |
| **Co-Attention**         | 查询/键/值维度 12 → 输出 12 | 同上                       |
| **Depthwise Conv Block** | Kernel = 3×3, 5×5, 7×7 × 各 64 通道<br>Pointwise → 64 → 拼接 → 192  | 同上                       |
| **SE Module**            | 通道数 192；压缩比 $16$    | 同上                       |
| **Transition Layer**     | BatchNorm → 1D Causal Conv<br>残差连接 | 同上                       |
| **Decoder(MLP)**         | FC (192 → 120) → ResBlock → FC (120 → 2×12) | 同上                       |

- **Top-k 值**：k=3

### 3.3 损失项权重、超参选择

- **lambda（PDL ）：10。  
- **（Logits 蒸馏权重）：0.5。  
- **温度：7。  
- **影响半径：5

### 3.4 训练细节

- **优化器**：Adam。  
  - 初始学习率 $lr = 10^-3；  
  - 每 100 轮衰减 10×；  
  - 总迭代轮数 300。  
- **批大小**：Batch = 8。  
- **梯度裁剪**：可选最大范数 5。  
- **Dropout**：0.1。  
- **硬件环境**：NVIDIA GeForce RTX 3060 Ti；  
- **框架版本**：  
  - PyTorch 1.13.1；  
  - TensorFlow 2.6.0；  
  - Python 3.8。  
