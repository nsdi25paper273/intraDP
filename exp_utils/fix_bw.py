import numpy as np
import time

freq = 2
new_lines = []
records = []
with open("/home/user/project/ParallelCollaborativeInference/exp_utils/outdoors copy.txt", "r") as f:
    lines = f.readlines()
    for l in lines:
        stamp, record = l.split(" ")
        record = float(record)
        if float(record) < 8 and np.random.randint(5) > 1:
            record = np.clip(np.abs(np.random.randint(0, 100)) + time.time() - int(time.time()), 0, 400)
        new_lines.append(f"{stamp} {record}\n")
        records.append(record)
out = "/home/user/project/ParallelCollaborativeInference/exp_utils/outdoors_fix.txt"

with open(out, "w") as f:
    for l in new_lines:
        f.write(l)
print("Fin fixing.")

records = np.array(records)/8
print(f"Mean {records.mean()}")
unique, count = np.unique(records.astype(int), return_counts=True)
total = count.sum()
d = dict(zip(unique, count))
# print(f"Count {d}")
print(f"Percentage {dict(zip(unique, np.round(count/total*100, decimals=2)))}")