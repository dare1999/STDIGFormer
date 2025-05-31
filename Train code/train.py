import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from processdata import get_train_testdata
import torch.nn.functional as F
from Net import MyNet, compute_projectdiffloss
from thop import profile

def compute_knowledge_distillation_loss(student_output, teacher_output, temperature=1):

    student_probs = F.log_softmax(student_output / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_output / temperature, dim=-1)

    assert student_probs.shape == teacher_probs.shape, \
        f"Shape mismatch: Student {student_probs.shape} vs Teacher {teacher_probs.shape}"

    kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return kd_loss


def train():
    historytimes = 8
    futuretimes = 12
    traindatax, trainadjm, traindatay, testdatax, testadjm, testdatay = get_train_testdata(
        historytimes=historytimes,
        futuretimes=futuretimes
    )

    for i in range(len(trainadjm)):
        for j in range(len(trainadjm[i])):
            if np.isnan(trainadjm[i][j]).any():
                print(i, j)

    epoches = 300
    batchsize = 8

    # 模型参数
    peoplenums = traindatax.shape[1]
    embed_size = 38
    inputdims = traindatax.shape[-1] + embed_size
    gatheadnums = 6
    gatprojectdims = 5
    gatlayernums = 2
    adjnums = len(trainadjm)
    transformerheadnums = 3
    t_transformerheadnums = 8
    t_gatheadnums = 6
    transformerprojectdims = 4
    transformerlayernums = 3
    t_gatlayernums = 4
    t_transformerlayernums = 6
    outputdims = traindatay.shape[2] * traindatay.shape[3]

    net = MyNet(
        peoplenums, embed_size, inputdims,
        gatheadnums, gatprojectdims, gatlayernums, adjnums,
        transformerheadnums, transformerprojectdims, transformerlayernums,
        outputdims
    )
    teacher_model = MyNet(
        peoplenums, embed_size, inputdims,
        t_gatheadnums, gatprojectdims, t_gatlayernums, adjnums,
        t_transformerheadnums, transformerprojectdims, t_transformerlayernums,
        outputdims
    )
    net.eval()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lambda_ = 10
    kd_weight = 0.5
    temperature = 7.0

    epoch_var_list = []

    for epoch in range(epoches):
        net.train()
        epoch_loss = 0.0
        epoch_var_sum = 0.0
        sample_count = 0

        for i in range(0, traindatax.shape[0], batchsize):
            batch_loss = 0.0
            batch_range = range(i, min(i + batchsize, traindatax.shape[0]))

            for j in batch_range:

                datax = torch.tensor(traindatax[j]).float()
                adjms = [torch.tensor(adj[j]).float() for adj in trainadjm]
                target = torch.tensor(traindatay[j]).float()

                student_output, graphnetlinearweights = net(datax, adjms, returnlinearweights=True)
                student_reshaped = student_output.view(target.shape)  # [T, N, F]
                feat_var = student_reshaped.var(dim=1).mean().item()

                epoch_var_sum += feat_var
                sample_count += 1

                if feat_var < 1e-5:
                    print(f"[Monitor] Epoch {epoch+1} Sample {j}: VERY LOW VAR = {feat_var:.6e}")

                # 教师模型输出
                with torch.no_grad():
                    teacher_output = teacher_model(datax, adjms)

                # 损失计算
                kd_loss = compute_knowledge_distillation_loss(
                    student_output.view(-1, outputdims),
                    teacher_output.view(-1, outputdims),
                    temperature
                )
                projectdiffloss = compute_projectdiffloss(graphnetlinearweights)
                mask = (target.sum(dim=-1) > 0).float()
                mseloss = torch.sum(
                    torch.mean((student_reshaped - target) ** 2, dim=-1) * mask
                ) / (torch.sum(mask) + 1e-10)

                total_loss = mseloss + lambda_ * projectdiffloss + kd_weight * kd_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_loss += total_loss.item()

            avg_batch_loss = batch_loss / len(batch_range)
            epoch_loss += avg_batch_loss
            print(f"Epoch {epoch+1}/{epoches} | Batch {i//batchsize} | Loss: {avg_batch_loss:.4f}")

        avg_epoch_var = epoch_var_sum / sample_count if sample_count > 0 else 0.0
        epoch_var_list.append(avg_epoch_var)
        print(f"Epoch {epoch+1}/{epoches} | Avg Node Feature Var: {avg_epoch_var:.6e} | Avg Loss: {epoch_loss/((traindatax.shape[0]//batchsize)+1):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(net, f"net_epoch_dis{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

    df = pd.DataFrame({
        'epoch': list(range(1, epoches+1)),
        'node_feature_variance': epoch_var_list
    })
    excel_path = 'feat_var_by_epoch_dis.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"Saved node feature variance per epoch to {excel_path}")

if __name__ == '__main__':
    train()