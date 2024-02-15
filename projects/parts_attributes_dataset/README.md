

## Collecting captions
Collecting attributes of parts about objects from LLM (chatgpt). This collection will be put in YAML file. 
```commandline
python projects/parts_attributes_dataset/extract_captions.py  --cfg  files/configs/ImageNet21k_training/ImageNet21K_training.yaml
```

Building traininga and testing data from the YAML file above. 
```commandline
 python projects/parts_attributes_dataset/create_json.py -i projects/parts_attributes_dataset/imagenet21k_parts_descriptions_train.yaml -k 10
```



## Training
Get openclip training environment:
```commandline
conda activate openclip
```

Run training: (current train set contains 455 unique image-text pairs, val set contains 74 unique image-text pairs)
```commandline
srun --gres=gpu:1 --partition=48gb python -m training.main \
    --save-frequency 5 \
    --zeroshot-frequency 5 \
    --report-to tensorboard \
    --train-data="projects/parts_attributes_dataset/imagenet21k_parts_descriptions_train.csv"  \
    --val-data="projects/parts_attributes_dataset/imagenet21k_parts_descriptions_val.csv"  \
    --csv-img-key "image_path" \
    --csv-caption-key "description" \
    --csv-separator "," \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=50 \
    --workers=8 \
    --logs="/shared-data5/guy/exps/grounding_clip_training" \
    --model ViT-L-14 \
    --pretrained='laion2b_s32b_b82k' 
```

Watch training:
```commandline
tensorboard --bind_all --logdir=/shared-data5/guy/exps/grounding_clip_training
```

## Testing
```commandline
srun --gres=gpu:1 --partition=24gb python projects/parts_attributes_dataset/test_itm.py \
    -m ViT-L-14 \
    -o /shared-data5/guy/exps/grounding_clip_training \
    -f projects/parts_attributes_dataset/CUB_attributes_gt_Ethan1.json \
    -c 'openai' \
    -p ""
```

```commandline
srun --gres=gpu:1 --partition=24gb python projects/parts_attributes_dataset/test_itm.py \
    -m ViT-L-14 \
    -o /shared-data5/guy/exps/grounding_clip_training \
    -f projects/parts_attributes_dataset/CUB_attributes_gt_Ethan1.json \
    -c 'laion2b_s32b_b82k' \
    -p ""
```

```commandline
srun --gres=gpu:1 --partition=24gb python projects/parts_attributes_dataset/test_itm.py \
    -p "" \
    -m ViT-L-14 \
    -o /shared-data5/guy/exps/grounding_clip_training \
    -f projects/parts_attributes_dataset/CUB_attributes_gt_Ethan1.json \
    -c /shared-data5/guy/exps/grounding_clip_training/2024_01_11-19_22_55-model_ViT-L-14-lr_0.0001-b_128-j_8-p_amp/checkpoints/epoch_50.pt 
```

## More trainings
```commandline
srun --gres=gpu:1 --partition=48gb python -m training.main \
    --save-frequency 5 \
    --zeroshot-frequency 5 \
    --report-to tensorboard \
    --train-data="projects/parts_attributes_dataset/imagenet21k_parts_descriptions_train.csv"  \
    --val-data="projects/parts_attributes_dataset/CUB_attributes_gt_Guy.csv"  \
    --csv-img-key "image_path" \
    --csv-caption-key "description" \
    --csv-separator "," \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=100 \
    --workers=8 \
    --logs="/shared-data5/guy/exps/grounding_clip_training" \
    --model ViT-L-14 \
    --pretrained='openai' 
```

## Local version for open_clip:
Install open_clip (user tag: v2.24.0, commit: 3ff1faf10b60be27252be7f6c84ce7c8c5e14ec8)
```commandline
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
```
Then,
```commandline
git checkout 3ff1faf10b60be27252be7f6c84ce7c8c5e14ec8
```
or
```commandline
git checkout tags/v2.24.0
```


Customize open_clip training:
```commandline
srun --gres=gpu:1 --partition=48gb python tools/train_openclip.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --val-frequency 1 \
    --class-attribute-val "files/configs/Dogs120/Dogs120_BaseAtrributeClassifier_L_14_finetuned.yaml" \
    --report-to tensorboard \
    --dataset-type 'csv_unique' \
    --train-data="projects/parts_attributes_dataset/imagenet21k_parts_descriptions_train_v2_200c.csv"  \
    --val-data="projects/parts_attributes_dataset/CUB_attributes_gt_Guy.csv"  \
    --csv-img-key "image_path" \
    --csv-caption-key "description" \
    --csv-separator "," \
    --warmup 10000 \
    --batch-size=64 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=1000 \
    --workers=8 \
    --logs="/shared-data5/guy/exps/grounding_clip_training" \
    --model ViT-L-14 \
    --pretrained='openai'
```

```commandline
srun --gres=gpu:1 --partition=24gb python projects/parts_attributes_dataset/test_itm.py \
    -p "" \
    -m ViT-L-14 \
    -o tmp \
    -f projects/parts_attributes_dataset/CUB_attributes_gt_Guy.json \
    -c /shared-data5/guy/exps/grounding_clip_training/2024_01_15-21_29_46-model_ViT-L-14-lr_0.0001-b_64-j_8-p_amp/checkpoints/epoch_2000.pt 
```


```commandline
srun --gres=gpu:2 --partition=24gb python tools/train_openclip.py \
    --save-frequency 10 \
    --zeroshot-frequency 10 \
    --val-frequency 10 \
    --class-attribute-val "files/configs/Dogs120/Dogs120_BaseAtrributeClassifier_L_14_finetuned.yaml" \
    --report-to tensorboard \
    --dataset-type 'csv_unique' \
    --train-data="projects/parts_attributes_dataset/imagenet21k_parts_descriptions_train_v1v2_371c.csv"  \
    --val-data="projects/parts_attributes_dataset/CUB_attributes_gt_Guy.csv"  \
    --csv-img-key "image_path" \
    --csv-caption-key "description" \
    --csv-separator "," \
    --warmup 10000 \
    --batch-size=32 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=1000 \
    --workers=8 \
    --logs="/shared-data5/guy/exps/grounding_clip_training" \
    --model ViT-L-14 \
    --pretrained='openai'
```

#### Fileter categories in ImageNet21k:
```commandline
python projects/parts_attributes_dataset/filter_imagenet21k_categories.py
```


#### Creating captions OX style:
```commandline
python tools/generate_descriptions_improved.py \
    --ann-path files/classnames/ImageNet21k/imagenet21k_filtered_classes_v1_10csingle_name.txt \
    --output-path tmp_2  \
    --description_reader list \
    -p description_improved -base_folder files/prompts/imagenet21k_get_raw_description -to_baseline_folder files/prompts/imagenet21k_from_raw_to_sentences  
```

```commandline
python projects/clip_training/convert_json_dataset_to_csv.py --file_path files/clip_training_data/ImageNet21k/baseline_descriptions.json --cfg_file files/configs/Dogs120/Dogs120_BaseAtrributeClassifier.yaml --out_dir files/clip_training_data/ImageNet21k/train_ox_v1
```



#### Jan 24-25:
```commandline
python projects/parts_attributes_dataset/filter_imagenet21k_categories.py -o files/classnames/ImageNet21k/imagenet21k_filtered_classes_man_made.txt -n 100000
```
```commandline
python tools/generate_descriptions.py -p ox_noname_imagenet21k -m gpt-3 -o files/descriptors/ImageNet21k/imagenet21k_filtered_classes_v4_7939.json -a files/classnames/ImageNet21k/imagenet21k_filtered_classes_v5.txt
```

```commandline
torchrun --nproc_per_node 1
```

Training with Ethan's data
```commandline
srun --gres=gpu:1 --partition=48gb python tools/train_openclip.py \
    --save-frequency 100000 \
    --zeroshot-frequency 10 \
    --val-frequency 10 \
    --class-attribute-val "files/configs/Dogs120/Dogs120_BaseAtrributeClassifier_L_14_finetuned.yaml" \
    --report-to tensorboard \
    --dataset-type 'csv_unique' \
    --train-data="files/clip_training_data/ImageNet1k/oxford_style/train.csv"  \
    --val-data="files/clip_training_data/ImageNet1k/oxford_style/test.csv"  \
    --csv-img-key "image_path" \
    --csv-caption-key "description" \
    --csv-separator "," \
    --warmup 10000 \
    --batch-size=64 \
    --lr=1e-5 \
    --wd=0.2 \
    --epochs=1000 \
    --workers=8 \
    --logs="/shared-data5/guy/exps/grounding_clip_training" \
    --model ViT-L-14 \
    --pretrained='openai' \
    --lock-text-unlocked-layers 10 \
    --lock-image-unlocked-groups 10 
```

```commandline
srun --gres=gpu:1 --partition=48gb python tools/train_openclip.py \
    --save-frequency 100000 \
    --zeroshot-frequency 10 \
    --val-frequency 10 \
    --class-attribute-val "files/configs/Dogs120/Dogs120_BaseAtrributeClassifier_L_14_finetuned.yaml" \
    --report-to tensorboard \
    --dataset-type 'csv_unique' \
    --train-data="files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v4_449c.csv"  \
    --val-data="files/clip_training_data/ImageNet1k/oxford_style/test.csv"  \
    --csv-img-key "image_path" \
    --csv-caption-key "description" \
    --csv-separator "," \
    --warmup 10000 \
    --batch-size=64 \
    --lr=1e-5 \
    --wd=0.2 \
    --epochs=1000 \
    --workers=8 \
    --logs="/shared-data5/guy/exps/grounding_clip_training" \
    --model ViT-L-14 \
    --pretrained='openai' \
    --lock-text-unlocked-layers 10 \
    --lock-image-unlocked-groups 10 
```
training file:
* files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v2v3_600c_plus_imagenet1K_no_dogs.csv
* files/clip_training_data/ImageNet1k/oxford_style/train.csv
* files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v4_449c.csv


#### Jan 28:
```commandline
python tools/generate_descriptions.py -p ox_noname_imagenet21k -m gpt-3 -o files/descriptors/ImageNet21k/imagenet21k_filtered_classes_28_01_5539c.json -a files/classnames/ImageNet21k/imagenet21k_filtered_classes_v5_not_in_df.txt
```


```commandline
torchrun --nproc_per_node 2 tools/train_openclip.py  \
srun --gres=gpu:1 --partition=48gb python tools/train_openclip.py  \
srun --gres=gpu:2 --partition=48gb python tools/train_openclip.py  \
    --save-frequency 10 \
    --zeroshot-frequency 10 \
    --val-frequency 10 \
    --class-attribute-val "files/configs/Dogs120/Dogs120_BaseAtrributeClassifier_L_14_finetuned.yaml" \
    --report-to tensorboard \
    --dataset-type 'csv_unique' \
    --val-data="files/clip_training_data/ImageNet1k/oxford_style/test.csv"  \
    --csv-img-key "image_path" \
    --csv-caption-key "description" \
    --csv-separator "," \
    --warmup 10000 \
    --batch-size=108 \
    --lr=1e-5 \
    --wd=0.1 \
    --epochs=1000 \
    --workers=8 \
    --logs="/shared-data6/guy/exps/grounding_clip_training/logs" \
    --model ViT-L-14 \
    --pretrained='openai' \
    --train-data="files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_28_01_2400c_plus_ethan.csv" \
    --name="2400c_plus_ethan_full_classes"   


    --train-data="files/clip_training_data/ImageNet1k/oxford_style/train.csv" \
    --name="ethan_full_classes"   
 
    
    
```
* files/clip_training_data/ImageNet1k/oxford_style/train.csv
* files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_26_01_2400c.csv
* files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_28_01_2400c_10KR_plus_ethan.csv
* files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_28_01_2400c_plus_ethan.csv

```commandline
torchrun --nproc_per_node 2 tools/train_openclip.py  \
    --save-frequency 100000 \
    --zeroshot-frequency 10 \
    --val-frequency 10 \
    --class-attribute-val "files/configs/Dogs120/Dogs120_BaseAtrributeClassifier_L_14_finetuned_full_classes.yaml" \
    --report-to tensorboard \
    --dataset-type 'csv_unique' \
    --val-data="files/clip_training_data/ImageNet1k/oxford_style/test.csv"  \
    --csv-img-key "image_path" \
    --csv-caption-key "description" \
    --csv-separator "," \
    --warmup 50 \
    --batch-size=512 \
    --lr=1e-5 \
    --wd=0.1 \
    --epochs=1000 \
    --workers=8 \
    --logs="/shared-data6/guy/exps/grounding_clip_training/logs" \
    --model ConvNext-Base \
    --pretrained='laion2b_s13b_b90k' \
    --train-data="files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_28_01_2400c_plus_ethan.csv" \
    --name="2400c_plus_ethan_ViT-B-32_bs512"
```

