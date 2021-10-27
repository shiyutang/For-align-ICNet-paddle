import sys
sys.path.append("..")
from models import *
import paddle
import numpy as np
import torch
from reprod_log import ReprodLogger


model_pd_file = 'model_92_sgd.pdparams'

model_pd = ICNet
# paddle.save(model_pd.state_dict(),'./tools/pdmodel.pdparams')
# torch.save(model.state_dict(), './tools/tmodel.pth')
model.load_state_dict((torch.load(model_file, map_location='cpu')))
model_pd.load_dict(paddle.load((model_pd_file)))

model.eval()
model_pd.eval()

fake_data = np.random.rand(64, 3, 32, 32).astype(np.float32)
output = model(torch.from_numpy(fake_data))
output2 = model_pd(paddle.to_tensor(fake_data))

reprod_logger = ReprodLogger()
reprod_logger_2 = ReprodLogger()

reprod_logger.add("logits", output.cpu().detach().numpy())
reprod_logger.save("forward_paddle.npy")

reprod_logger_2.add("logits", output2.cpu().detach().numpy())
reprod_logger_2.save("forward_torch.npy")