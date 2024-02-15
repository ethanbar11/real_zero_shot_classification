# Documentation of code work and experiments for ICML 2024 paper:

### Creating data:
##### Creating data for Columbia style on ImageNet21k:
```commandline
python tools/generate_descriptions.py -p col \
    -a files/clip_training_data/golden_dataset/ox_all_including_man_made200_classes.txt \
    -o files/clip_training_data/golden_dataset/col_all_including_man_made200.json
```

##### Creating data for Oxford style on ImageNet21k:
```commandline
python tools/generate_descriptions.py -p ox_noname_imagenet21k \
    -a files/clip_training_data/golden_dataset/ox_all_including_man_made200_classes.txt \
    -o files/clip_training_data/golden_dataset/col_all_including_man_made200.json
```

##### Creating data for Columbia style on Cars196:
```commandline
python tools/generate_descriptions.py -p col \
    -a files/classnames/Cars196/cars196.txt \
    -o files/descriptors/Cars196/cars196_gpt3_text-davinci-003_descriptions_col.json
```
Creating description files with class_names, and class_names_only:
```commandline
python tools/add_names_to_descriptions.py -f files/descriptors/Cars196/cars196_gpt3_text-davinci-003_descriptions_col.json
```

##### Creating data for Oxford style on Cars196:
```commandline
python tools/generate_descriptions.py -p ox_noname_imagenet21k \
    -a files/classnames/Cars196/cars196.txt \
    -o files/descriptors/Cars196/cars196_gpt3_text-davinci-003_descriptions_ox_noname.json
```
Creating description files with class_names, and class_names_only:
```commandline
python tools/add_names_to_descriptions.py -f files/descriptors/Cars196/cars196_gpt3_text-davinci-003_descriptions_ox_noname.json
```


##### Creating data for Columbia style on CUB:
```commandline
python tools/generate_descriptions.py -p col \
    -a files/classnames/CUB/cub.txt \
    -o files/descriptors/CUB/cub_gpt3_text-davinci-003_descriptions_col.json
```
Creating description files with class_names, and class_names_only:
```commandline
python tools/add_names_to_descriptions.py -f files/descriptors/CUB/cub_gpt3_text-davinci-003_descriptions_col.json
```

##### Creating data for Columbia style on Dogs120:
```commandline
python tools/generate_descriptions.py -p col \
    -a files/classnames/Dogs120/dogs120.txt \
    -o files/descriptors/Dogs120/cub_gpt3_text-davinci-003_descriptions_col.json
```
Creating description files with class_names, and class_names_only:
```commandline
python tools/add_names_to_descriptions.py -f files/descriptors/Dogs120/cub_gpt3_text-davinci-003_descriptions_col.json
```

##### Creating data for Columbia style on FLOWERS102:
```commandline
python tools/generate_descriptions.py -p col \
    -a files/classnames/FLOWERS102/flowers102.txt \
    -o files/descriptors/FLOWERS102/flowers102_gpt3_text-davinci-003_descriptions_col.json
```
Creating description files with class_names, and class_names_only:
```commandline
python tools/add_names_to_descriptions.py -f files/descriptors/FLOWERS102/flowers102_gpt3_text-davinci-003_descriptions_col.json
```

##### Creating data for Columbia style on Food101:
```commandline
python tools/generate_descriptions.py -p col \
    -a files/classnames/Food101/all.txt \
    -o files/descriptors/Food101/food_gpt3_text-davinci-003_descriptions_col.json
```
Creating description files with class_names, and class_names_only:
```commandline
python tools/add_names_to_descriptions.py -f files/descriptors/Food101/food_gpt3_text-davinci-003_descriptions_col.json
```

##### Creating data for Columbia style on OxfordPets:
```commandline
python tools/generate_descriptions.py -p col \
    -a files/classnames/OXFORD_PET/oxford_pet.txt \
    -o files/descriptors/OXFORD_PET/oxford_pet_gpt3_text-davinci-003_descriptions_col.json
```
Creating description files with class_names, and class_names_only:
```commandline
python tools/add_names_to_descriptions.py -f files/descriptors/OXFORD_PET/oxford_pet_gpt3_text-davinci-003_descriptions_col.json
```





### Experiments for deficiencies of CLIP
Experiments for Oxford style are based on the following json description files:

##### Cars196
```commandline
files/classnames/Cars196/cars196.txt
files/descriptors/Cars196/cars196_gpt3_text-davinci-003_descriptions_ox_noname_filtered.json
files/descriptors/Cars196/cars196_gpt3_text-davinci-003_descriptions_ox_noname_filtered_with_classname.json.
files/descriptors/Cars196/cars196_gpt3_text-davinci-003_descriptions_ox_noname_filtered_with_classname_only.json.
```

##### CUB
```commandline
files/classnames/CUB/cub.txt
files/descriptors/CUB/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4.json
files/descriptors/CUB/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4_with_classname.json.
files/descriptors/CUB/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4_with_classname_only.json.
```

##### Dogs120
```commandline
files/classnames/Dogs120/dogs120.txt
files/descriptors/Dogs120/descriptions_for_real_zs.json
files/descriptors/Dogs120/descriptions_for_real_zs_with_classname.json
files/descriptors/Dogs120/descriptions_for_real_zs_with_classname_only.json
```

##### FLOWERS102
```commandline
files/classnames/FLOWERS102/flowers102.txt
files/descriptors/FLOWERS102/flowers102_gpt3_text-davinci-003_descriptions_ox_noname_clean.json
files/descriptors/FLOWERS102/flowers102_gpt3_text-davinci-003_descriptions_ox_noname_clean_with_classname.json.
files/descriptors/FLOWERS102/flowers102_gpt3_text-davinci-003_descriptions_ox_noname_clean_with_classname_only.json.
```

##### Food101
```commandline
files/classnames/Food101/all.txt
files/descriptors/Food101/food_gpt3_text-davinci-003_descriptions_ox_noname_filtered.json
files/descriptors/Food101/food_gpt3_text-davinci-003_descriptions_ox_noname_filtered_with_classname.json.
files/descriptors/Food101/food_gpt3_text-davinci-003_descriptions_ox_noname_filtered_with_classname_only.json.
```

##### OxfordPets
```commandline
files/classnames/OXFORD_PET/oxford_pet.txt
files/descriptors/OXFORD_PET/oxford_pet_gpt3_text-davinci-003_descriptions_ox_noname.json
files/descriptors/OXFORD_PET/oxford_pet_gpt3_text-davinci-003_descriptions_ox_noname_with_classname.json.
files/descriptors/OXFORD_PET/oxford_pet_gpt3_text-davinci-003_descriptions_ox_noname_with_classname_only.json.
```