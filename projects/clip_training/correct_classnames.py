json_path = r'../../files/descriptors/FLOWERS102/flowers102_gpt3_text-davinci-003_descriptions_ox_noname_clean.json'
import json
with open(json_path, 'r') as f:
    data = json.load(f)

classnames = list(data.keys())
class_out_path = r'../../files/classnames/FLOWERS102/flowers102.txt'
with open(class_out_path, 'w') as f:
    for c in classnames:
        f.write(c+'\n')