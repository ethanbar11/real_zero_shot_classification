
Firstly, create descriptions file with:
```commandline
python tools/generate_descriptions_improved.py --ann-path files/classnames/OXFORD_PET/opensets/twocls/unseen.txt -p /
 description_improved -base_folder files/prompts/oxford_pets_get_raw_description /
 -to_baseline_folder files/prompts/oxford_pets_from_raw_to_sentences -o tmp_2
```
Notice there is a 2-step prompt curation happening here. Firstly,
we use the prompts from base_folder to create the description of the object (with its name),
and secondly we use the prompts from to_baseline_folder to split it into coherent sentences,
and remove the name.

Afterwards, use convert_json_dataset_to_csv.py to create the csv file for training:

```commandline
python projects/clip_training/convert_json_dataset_to_csv.py \
 --file_path files/clip_training_data/Dogs120/baseline_descriptions.json \
 --cfg_file files/configs/Dogs120/Dogs120_BaseAtrributeClassifier.yaml \
 --out_dir files/clip_training_data/Dogs120/different_class_split
 --description_reader list
```

Notice it saves the train and test.txt that save the class split, and train.csv and test.csv for the CLIP training.
It also save the descriptions that created the csvs in the same directory.

Now for the CLIP training:


```commandline
srun --gres=gpu:1 --partition=48gb python -m projects.parts_attributes_dataset.train_openclip  --save-frequency 100  \
     --zeroshot-frequency 100     --report-to tensorboard     --dataset-type 'csv_unique' \
     --train-data="files/clip_training_data/Dogs120/different_class_split/train.csv"      \
     --val-data="files/clip_training_data/Dogs120/different_class_split/test.csv"      --csv-img-key "image_path" \
     --csv-caption-key "description"     --csv-separator ","     --warmup 10000     --batch-size=64     --lr=1e-3 \
     --wd=0.0     --epochs=1000     --workers=8     --logs="/shared-data5/ethan_baron/experiments/clip_training"  \
     --model ViT-L-14     --pretrained='openai' --name "Dogs orginal prompt new class split"
```