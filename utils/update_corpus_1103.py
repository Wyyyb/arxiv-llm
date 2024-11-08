import json


data = []
with open("../corpus_data/meta_data_1022.jsonl", "r") as fi:
    for line in fi.readlines():
        curr = json.loads(line)
        curr["abstract"] = curr["abstract"].replace("<|reference_start|>", "<|reference_start|> ")\
            .replace("<|reference_end|>", " <|reference_end|>")
        data.append(curr)

print("length of meta data", len(data))
with open("../corpus_data/meta_data_1108.jsonl", "w") as fo:
    for each in data:
        fo.write(json.dumps(each))
        fo.write("\n")



