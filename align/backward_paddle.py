import paddle
import paddle.nn as nnpd
from model.ran_paddle import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
import torch
import torch.nn as nn
import numpy as np
from reprod_log import ReprodLogger
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModeltorch
import torch.optim as optim
import paddle.optimizer as optimpd
if __name__ == "__main__":
    lr = 0.1
    fake_data = np.load("./fake_data.npy")
    fake_label = np.load("./fake_label.npy")
    print(fake_data)
    print(fake_label)
    model_file = 'model_92_sgd.pkl'
    model_pd_file = 'model_92_sgd.pdparams'
    paddle.set_device("cpu")
    # def logger
    reprod_logger1 = ReprodLogger()
    reprod_logger2 = ReprodLogger()

    # paddle
    fake_datapd = paddle.to_tensor(fake_data)
    fake_labelpd = paddle.to_tensor(fake_label)
    model_pd = ResidualAttentionModel()
    model_pd.load_dict(paddle.load((model_pd_file)))
    model_pd.eval()
    criterionpd = nnpd.CrossEntropyLoss()
    optimizerpd = optimpd.Momentum(parameters=model_pd.parameters(), momentum=0.9, use_nesterov=True, weight_decay=0.0001)
    # 读取fake data和fake label
    # forward
    optimizerpd.clear_grad()
    out = model_pd(fake_datapd)
    losspd = criterionpd(out, fake_labelpd)
    losspd.backward()
    optimizerpd.step()
    # 计算loss的值
    losspd = criterionpd(out, fake_labelpd)
    # 记录loss到文件中
    reprod_logger1.add("backward", losspd.cpu().detach().numpy())
    reprod_logger1.save("backward_paddle.npy")


    # torch
    torch.device("cpu")
    # def logger
    fake_datat = torch.from_numpy(fake_data)
    fake_labelt = torch.from_numpy(fake_label)
    model = ResidualAttentionModeltorch()
    model.load_state_dict((torch.load(model_file, map_location='cpu')))
    model.eval()
    criteriont = nn.CrossEntropyLoss()
    optimizert = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    # 读取fake data和fake label
    # forward
    optimizert.zero_grad()
    outt = model(fake_datat)
    losst = criteriont(outt, fake_labelt)
    losst.backward()
    optimizert.step()
    # 计算loss的值
    losst = criteriont(outt, fake_labelt)
    # 计算loss的值
    # 记录loss到文件中
    reprod_logger2.add("backward", losst.cpu().detach().numpy())
    reprod_logger2.save("backward_torch.npy")
