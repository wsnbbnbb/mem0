from json import JSONDecodeError
import json
import os
import time
from collections import defaultdict
with open("/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo10.json", "r") as f:
            data = json.load(f)

with open("/root/nfs/hmj/proj/mem0/evaluation/src/dataset/lcm_test.json", "r") as f1:
            data1 = json.load(f1)
data1[0]["qa"]=data[9]["qa"]
print(data1[0]["qa"][0])
with open("/root/nfs/hmj/proj/mem0/evaluation/src/dataset/lcm_test.json", "w") as f2:
     json.dump(data1, f2, indent=4)