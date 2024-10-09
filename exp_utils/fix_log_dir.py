import os, shutil
from os import path as osp

path = "/project/ParallelCollaborativeInference/log"

for item in sorted(os.listdir(path)):
    if "local" in item:
        splits = item.split("_")
        splits[0] = splits[-1]
        new_name = "_".join(splits[:-1])
        shutil.move(osp.join(path, item), osp.join(path, new_name))