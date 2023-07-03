import os

# os.environ['PT_XLA_DEBUG'] = '1' 
# os.environ['PT_XLA_DEBUG_FILE'] = './xla_debug.txt'
os.environ['XLA_IR_DEBUG'] = '1'
os.environ['XLA_HLO_DEBUG'] = '1'
# os.environ['XLA_EMIT_STEPLOG'] = '1'
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['XLA_USE_BF16'] = '1'
os.environ['MASTER_PORT'] = '29510'

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt as pjrt
import torch_xla.debug.profiler as xp
import cv2
import time
import multiprocessing
import pickle
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def trace_fn():
    time.sleep(120)
    print('++++++++ START OF PROFILING ++++++++')
    xp.trace(
        'localhost:6009', 
        logdir='/home/lodestone/sam-hq/',  
        num_tracing_attempts=3, 
        host_tracer_level=3, 
        timeout_s=15, 
        duration_ms=30000)
    print('++++++++  END OF PROFILING  ++++++++')
trace = False
p = multiprocessing.Process(target=trace_fn)

SERIAL_EXEC = xmp.MpSerialExecutor()

def _mp_fn(index):
    server = xp.start_server(6009)

    print(f"[xla:{xm.get_ordinal()}] spawned process")
    # Per-device setup
    import torch_xla.experimental.pjrt_backend
    dist.init_process_group('xla', init_method='pjrt://')
    device = xm.xla_device()
    
    # Model loading
    sam_checkpoint = "pretrained_checkpoint/sam_hq_vit_h.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    print(f"[xla:{xm.get_ordinal()}] model moved to device")

    sam.eval()
    sam_dynamo = torch.compile(sam, backend='torchxla_trace_once')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_dynamo, # type: ignore
        points_per_side=8,
        points_per_batch=64
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
            filename = self.file_list[idx]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)
            return image, filename
    
    class HWCtoCHWTransform():
        def __call__(self, img):
            return torch.as_tensor(img).permute(2, 0, 1)

    def load_dataset():
        data_directory = "./data/zd_testimgs/"
        #data_directory = "./one_testimg/"
        #data_directory = "./two_testimgs/"
        return CustomImageDataset(
            data_directory,
            transform=HWCtoCHWTransform())
    
    dataset = SERIAL_EXEC.run(lambda: load_dataset())
    generation_sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1, 
        sampler=generation_sampler, 
        num_workers=0)
    generation_loader = pl.MpDeviceLoader(dataloader,device)

    print(f"[xla:{xm.get_ordinal()}] dataset loaded")
    tracker = xm.RateTracker()
    from PIL import Image

    # Main processing loop
    for batch_idx, (input, filename) in enumerate(generation_loader):
        masks, valid = mask_generator.generate(input)
        tracker.add(1)
        validcount = valid.to(torch.int8).count_nonzero().cpu().item()
        print(f"[xla:{xm.get_ordinal()}] batch {batch_idx}, got {validcount} masks. Rate {tracker.rate()}, global {tracker.global_rate()}")

        # Convolution operation for postprocessing masks
        with xp.Trace('postprocess_conv'):
            if validcount > 0:
                random_index = torch.randint(0, int(validcount), (1,))
                kernel = torch.tensor(
                    [[[[0,1,1,1,0],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [0,1,1,1,0]]]],
                    dtype=torch.bool, 
                    device=xm.xla_device()
                )
                mask_convolving = masks[0][random_index]
                mask_convolving = ~mask_convolving
                for i in range(1): # Erode for one step to clean up areas
                    mask_convolving = torch.nn.functional.conv2d(mask_convolving, kernel, padding=(2,2))
                mask_convolving = ~mask_convolving
                for i in range(6): # Dilate for 10 steps to simulate inpainting style mask
                    mask_convolving = torch.nn.functional.conv2d(mask_convolving, kernel, padding=(2,2))
            else:
                mask_convolving = torch.ones_like(masks[0][0].unsqueeze(0), device=xm.xla_device())
            mask_convolving = mask_convolving.squeeze(0).squeeze(0)
            mask_convolving = torch.where(mask_convolving, torch.tensor(255, dtype=torch.int8, device=xm.xla_device()), torch.tensor(0, dtype=torch.int8, device=xm.xla_device()))
            image = Image.fromarray(mask_convolving.cpu().numpy(), mode='L')
            image.save(os.path.join('./data/out_masks/', filename[0]))


if __name__ == '__main__':
    if trace: p.start()
    # _mp_fn(1)
    xmp.spawn(_mp_fn, args=(), start_method='spawn')