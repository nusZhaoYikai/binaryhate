import random
import os
import numpy as np
import torch

def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # 速度会变慢，可以防止 LSTM 报错
    torch.backends.cudnn.enabled = False

