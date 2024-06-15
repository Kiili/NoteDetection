import gc
import math
import psutil

import torch
from torch.utils.data import DataLoader
from mingus.containers import Note


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def get_vram_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    t /= 1000000000
    f /= 1000000000
    return f, t


def get_ram_usage():
    return psutil.virtual_memory()[2] / 100


def int_to_note(key, velocity=100):
    return Note(name=(key+9), velocity=velocity)


def find_batch_size(model, dataset, max_val=math.inf, is_train=True, device=torch.device("cpu"), **dl_args):
    model.train(is_train)
    dl = DataLoader(dataset, **dl_args, batch_size=1)
    outputs = model(next(iter(dl))[0].to(device))
    del outputs
    del dl
    empty_cache()
    def do_try(batch_size):
        if batch_size > len(dataset):
            return False
        dl = DataLoader(dataset, **dl_args, batch_size=batch_size)
        try:
            if is_train:
                outputs = model(next(iter(dl))[0].to(device))
            else:
                with torch.no_grad():
                    outputs = model(next(iter(dl))[0].to(device))
            del outputs
            del dl
            empty_cache()
            return True
        except:
            empty_cache()
            return False

    end = 1
    while do_try(end):
        if end > max_val:
            return min(max_val, len(dataset))
        end *= 2

    start = end // 2
    while start < end:
        mid = (start + end) // 2
        if do_try(mid):
            start = mid + 1
        else:
            end = mid - 1
    model.zero_grad()
    return min(max_val, len(dataset)) if end >= len(dataset) else max(1, min(len(dataset), int(max_val), end - 2, int(end * 0.8)))
