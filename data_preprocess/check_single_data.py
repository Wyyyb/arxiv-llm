import json


input_file = "../data/train_data_1016_0.jsonl"
output_file = "../local/check_single_data.json"
row_id = 376

data = []
with open(input_file, "r") as fi:
    for line in fi.readlines():
        curr = json.loads(line)
        data.append(curr)

res = data[row_id: row_id + 2]

with open(output_file, "w") as fo:
    fo.write(json.dumps(res))





