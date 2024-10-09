import os, shutil
from os import path as osp

path = "/project/ParallelCollaborativeInference/log"
fix_path = "/project/ParallelCollaborativeInference/old_log"
num = 3

print(f"Items that have problem:")
for item in sorted(os.listdir(path)):
    _path = osp.join(path, item)
    _items = os.listdir(_path)
    if len(_items) != num:
        assert "power_record.txt" not in _items
        if "kapao" not in item and "agrnav" not in item:
            split = item.split("_")
            split.insert(2, "torchvision")
            item = "_".join(split)
        fix_power_record_path = osp.join(fix_path, item, "power_record.txt")
        try:
            shutil.copy(fix_power_record_path, _path)
            print(f"Fix power record for {_path}")
        except:
            print(f"Error {_path}")
        
    