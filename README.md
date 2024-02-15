# *Real* Zero-Shot Classification

Our goal is to classify images of categories of fine objects, each category being described 
by a text description that does not contain evidence of the object's name in it (i.e., real zero-shot classification).

# CLIP Deficiencies
Firstly, we examine how good is CLIP with attributes. more info in the directory attribute_tests.

# Finetuning CLIP
To finetune CLIP with oxford style prompts, we use the description file infiles/clip_training_data/golden_dataset/ox_all_including_man_made200_filtered_total_classes.json,
and the class list files/clip_training_data/golden_dataset/ox_all_including_man_made200_filtered_total_classes_classes.txt.
To create a csv you can use the following format:
```commandline
python projects/parts_attributes_dataset/json2csv.py \
    --a files/clip_training_data/golden_dataset/ox_all_including_man_made200_filtered_total_classes.json \
    -c files/clip_training_data/golden_dataset/ox_all_including_man_made200_filtered_total_classes_classes.txt \
    -o files/clip_training_data/golden_dataset/train.csv \
    -m 20
```
Will create a csv file containing 20 images of each of the classes multiplied by the description amount of the class.

Then, to run the training itself:

```commandline
srun --gres=gpu:1 --partition=48gb python  tools/train_openclip.py \
    --save-frequency 10 \
    --zeroshot-frequency 10 \
    --val-frequency 10 \
    --class-attribute-val "files/configs/OXFORD_PET/OXFORD_PET_BaseAtrributeClassifier_finetuned_base.yaml" \
    --report-to tensorboard \
    --dataset-type csv_unique \
    --train-data="files/clip_training_data/golden_dataset/train.csv"  \
    --val-data="files/clip_training_data/ImageNet1k/oxford_style/test.csv"  \
    --csv-img-key "image_path" \
    --csv-caption-key "description" \
    --csv-separator "," \
    --warmup 50 \
    --batch-size=400 \
    --lr=1e-6 \
    --wd=0.5 \
    --epochs=150 \
    --workers=8 \
    --logs="/home/tmp/experiment_results/grounding_oxford_pet" \
    --model ViT-B-16 \
    --pretrained=openai \
```
