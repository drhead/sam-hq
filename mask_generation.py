import os

# os.environ['PT_XLA_DEBUG'] = '1' 
# os.environ['PT_XLA_DEBUG_FILE'] = './xla_debug.txt'
# os.environ['XLA_IR_DEBUG'] = '1'
# os.environ['XLA_HLO_DEBUG'] = '1'
# os.environ['XLA_EMIT_STEPLOG'] = '1'

os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['XLA_USE_BF16'] = '1'
os.environ['MASTER_PORT'] = '29510'

import numpy as np
import torch
print("PyTorch version:", torch.__version__)
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt as pjrt
import torch_xla.debug.profiler as xp
import cv2
import time
import multiprocessing
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def trace_fn():
    xp.trace('localhost:6009', logdir='/home/lodestone/sam-hq/', num_tracing_attempts=1, host_tracer_level=3, timeout_s=15, duration_ms=10000)
    print('Done tracing')
trace = False
p = multiprocessing.Process(target=trace_fn)
if trace:
    p.start()
    server = xp.start_server(6009)
    time.sleep(5)

SERIAL_EXEC = xmp.MpSerialExecutor()

def _mp_fn(index):
    print(f"[xla:{xm.get_ordinal()}] spawned process")
    # Per-device setup
    import torch_xla.experimental.pjrt_backend
    dist.init_process_group('xla', init_method='pjrt://')
    device = xm.xla_device()

    # Model loading
    sam_checkpoint = "pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    print(f"[xla:{xm.get_ordinal()}] model moved to device")
    pjrt.broadcast_master_param(sam)
    sam.eval()
    sam_dynamo = torch.compile(sam, backend='torchxla_trace_once')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_dynamo, # type: ignore
        points_per_side=4,
        points_per_batch=16
    )
    print(f"[xla:{xm.get_ordinal()}] model loaded")

    # Dataset loading
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
            # print(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)
            return image
    
    def load_dataset():
        data_directory = "./data/zd_testimgs/"
        return CustomImageDataset(data_directory)
    
    dataset = SERIAL_EXEC.run(lambda: load_dataset())
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=4, 
        sampler=train_sampler, 
        num_workers=0)
    train_loader = pl.MpDeviceLoader(dataloader,device)
    print(f"[xla:{xm.get_ordinal()}] dataset loaded")
    tracker = xm.RateTracker()

    # Main processing loop
    for batch_idx, (input) in enumerate(train_loader):
        tic = time.perf_counter()
        print(f"[xla:{xm.get_ordinal()}] starting batch {batch_idx}")
        input=input.cpu().numpy()
        for i in range(input.shape[0]):
            masks = mask_generator.generate(input[i], multimask_output=False)
            tracker.add(1)
            print(f"[xla:{xm.get_ordinal()}] batch {batch_idx}, got {masks.shape[0]} masks. Rate {tracker.rate()}, global {tracker.global_rate()}")
        toc = time.perf_counter()
        print(f"[xla:{xm.get_ordinal()}] Processed batch in {toc - tic:0.4f} seconds")

if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(), start_method='spawn')