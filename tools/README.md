# Tools for OV-OD


####
#### Generate descriptions
####
Generate descriptions for a given set of attributes, using a given model.
```commandline
python tools/generate_descriptions.py -a files/classnames/shoes_boots.txt -p col -m text-davinci-003
python tools/generate_descriptions.py -a /shared-data5/guy/data/lvis/lvis_v1_val.json -p col -m text-davinci-003
python tools/generate_descriptions.py -a files/classnames/metal_manmade.txt -p col -m text-davinci-003
python tools/generate_descriptions.py -a files/classnames/FLOWERS102/opensets/tiny/unseen.txt -m text-davinci-003 -o files/descriptors/{}_gpt3_{}_descriptions_{}.json
python tools/generate_descriptions.py -a files/classnames/FLOWERS102/flowers102.txt -m text-davinci-003 -o files/descriptors/{}_gpt3_{}_descriptions_{}.json -p ox_noname
python tools/generate_descriptions.py -a files/classnames/CUB/cub.txt -m text-davinci-003 -o files/descriptors/{}_gpt3_{}_descriptions_{}.json -p ox_noname
```

####
#### Create prompt
####
Create prompt for a given set of attributes, using a given model.
```commandline
python tools/create_prompt.py -o files/programs/set6/ -n "Rhinoceros Auklet" -p type2
python tools/create_prompt.py -d files/descriptors/attributes/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4.json -p type2 -o files/programs/set6/ -n "Rhinoceros Auklet"
python tools/create_prompt.py -d files/descriptors/attributes/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4.json -p type2 -o files/programs/set6/ -n "Sooty Albatross" 
```


####
#### Generate programs
####
Generate programs for a given set of descriptions, using a given openai chat-gpt model.
See [here](../README.md) for examples.


####
#### Pretty print json
####
```commandline
python tools/prettify_json.py -j files/descriptors/attributes/cub_gpt3_text-davinci-003_descriptions_col_prompt.json
```

Pretty print json with a given prefix:
```commandline
python tools/prettify_json.py -j files/descriptors/attributes/cub_gpt3_text-davinci-003_descriptions_col_prompt.json -p "A bird with "
```

####
#### Show data:
####
```commandline
python tools/show_data.py --cfg files/configs/CUB/DEBUG.yaml
```

####
#### Show optimization search space:
####
```commandline
python tools/show_search_graph.py -r tmp3/CUB/threecls/BaseProgramClassifierV2/ViT_L_14_336px/files_programs_CUB_ethan_23_10_23/30-10-2023_15-41-34/programs/
```


### Generate synthetic images from descriptions
