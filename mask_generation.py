import os

os.environ['PT_XLA_DEBUG'] = '1' 
os.environ['PT_XLA_DEBUG_FILE'] = './xla_debug.txt'
os.environ['XLA_IR_DEBUG'] = '1'
os.environ['XLA_HLO_DEBUG'] = '1'
os.environ['XLA_EMIT_STEPLOG'] = '1'

os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['XLA_USE_BF16'] = '1'
os.environ['MASTER_PORT'] = '29500'

import numpy as np
import torch
print("PyTorch version:", torch.__version__)
import torch_xla.core.xla_model as xm

if xm.get_xla_supported_devices("GPU") is not None:
    print("XLA GPU is available")
elif xm.get_xla_supported_devices("TPU") is not None:
    print("XLA TPU is available:", xm.get_xla_supported_devices("TPU",8))

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.distributed as dist
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt
import matplotlib.pyplot as plt
import cv2
import time
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu
import multiprocessing
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

dist.init_process_group('xla', init_method='pjrt://')

from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.file_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_list[idx])
        image = cv2.imread(img_path)
        print(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image
    
data_directory = "./data/zd_testimgs/"
dataset = CustomImageDataset(data_directory)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

sam_checkpoint = "pretrained_checkpoint/sam_hq_vit_h.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
WRAPPED_MODEL = xmp.MpModelWrapper(sam)

def gen_masks():
    xp.trace('localhost:6009', logdir='/home/lodestone/sam-hq/', num_tracing_attempts=1, host_tracer_level=3, timeout_s=15, duration_ms=10000)
    print('Done tracing')
trace = False
p = multiprocessing.Process(target=gen_masks)
if trace:
    p.start()
    server = xp.start_server(6009)
    time.sleep(5)

SERIAL_EXEC = xmp.MpSerialExecutor()
def load_dataset():
    return dataloader
    # dataset = CustomImageDataset(path)
    # return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

def _mp_fn(index, model):
    print(f"Spawned process for index {index}")
    dist.init_process_group('xla', init_method='pjrt://')
    model = WRAPPED_MODEL.to(xm.xla_device())
    print(f"Model moved to device on {index}")
    model.eval()
    sam_dynamo = torch.compile(model, backend='torchxla_trace_once')
    print(f"Torch compiled on {index}")
    # pjrt.broadcast_master_param(sam_dynamo)
    ddp_model = DDP(model, gradient_as_bucket_view=True)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_dynamo, # type: ignore
        points_per_side=4,
        points_per_batch=16
    )
    print(f"Model loaded on {index}")
    train_loader = SERIAL_EXEC.run(lambda: load_dataset())
    print(f"Dataset loaded on {index}")
    for batch_idx, (input) in enumerate(train_loader):
        print(f"On device {index}, starting batch {batch_idx}")

        for i in range(input.shape[0]):
            masks = mask_generator.generate(input[i, ...].numpy(), multimask_output=False)
            print(f"On device {index}, batch {batch_idx}, got {masks.shape[0]} masks")

if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(sam, ), start_method='fork')