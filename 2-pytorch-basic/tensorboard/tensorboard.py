# pip install tensorboard
# from torch.utils.tensorboard import SummaryWriter
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 以上两行代码为了解决： OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter

writer = SummaryWriter("logs")
# 注意名字不能相同
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

image_path = "/2-pytorch-basic/tensorboard/tensorboard_data/train/ants/0013035.jpg"
image_array = np.array(Image.open(image_path))
writer.add_image("image", image_array, 2, dataformats='HWC')
#
writer.close()
# 运行命令，进入网页查看
# tensorboard --logdir=logs --port=6007
