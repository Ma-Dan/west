import json
import re
import os

path = "aishell3"
path_list = os.listdir(path)
path_list.sort()

path_list = path_list[:-5]

dataset = []

for p in path_list:
    file_name = path + "/" + p + "/prosody/prosody.txt"
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()

    for i in range(len(lines)//2):
        d = lines[i*2].split('\t')
        dataset.append({
            "wav": path + "/" + p + "/wav/" + d[0] + ".wav",
            "txt": re.sub(r'#|[0-9]+', '', d[1]).rstrip("\n")
        })

#print(dataset)
f = open("train.jsonl", 'w')

for d in dataset:
    f.write(json.dumps(d, ensure_ascii=False) + '\n')

f.close()