
import json
path = r'files/descriptors/Food101/food_gpt3_text-davinci-003_descriptions_ox_noname.json'
with open(path,'r') as f:
    data = json.load(f)

new_data= {}
for name, atts in data.items():
    new_atts = []
    for att in atts:
        # Check that attrbiute doesn't contain numbers
        if len(att) < 200 and not any(char.isdigit() for char in att):
            new_atts.append(att.replace('animal','food'))
    new_data[name] = new_atts
out_path = r'../../files/descriptors/Food101/food_gpt3_text-davinci-003_descriptions_ox_noname_filtered.json'
with open(out_path,'w') as f:
    json.dump(new_data,f,indent=4)