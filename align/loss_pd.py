import paddle
import paddle.nn as nnpd
from model.ran_paddle import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
import torch
import torch.nn as nn
import numpy as np
from reprod_log import ReprodLogger
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModeltorch
if __name__ == "__main__":
    lr = 0.1
    # paddle
    model_file = 'model_92_sgd.pkl'
    model_pd_file = 'model_92_sgd.pdparams'
    paddle.set_device("cpu")
    # def logger
    reprod_logger = ReprodLogger()
    model_pd = ResidualAttentionModel()
    model_pd.load_dict(paddle.load((model_pd_file)))
    model_pd.eval()

    criterion = nnpd.CrossEntropyLoss()

    # 读取fake data和fake label
    fake_data = np.load("./fake_data.npy")
    fake_label = np.load("./fake_label.npy")

    fake_datapd = paddle.to_tensor(fake_data)
    fake_labelpd = paddle.to_tensor(fake_label)
    # forward
    out = model_pd(fake_datapd)
    # 计算loss的值

    loss = criterion(out, fake_labelpd)
    # 记录loss到文件中
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_paddle.npy")


    # torch
    torch.device("cpu")
    fake_datat = torch.from_numpy(fake_data)
    fake_labelt = torch.from_numpy(fake_label)
    # def logger
    reprod_logger = ReprodLogger()
    model = ResidualAttentionModeltorch()
    model.load_state_dict((torch.load(model_file, map_location='cpu')))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    # 读取fake data和fake label

    # forward
    out = model(fake_datat)
    # 计算loss的值
    loss = criterion(out, fake_labelt)
    # 记录loss到文件中
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")