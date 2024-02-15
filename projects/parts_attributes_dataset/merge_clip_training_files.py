import pandas as pd

def merge_clip_training_files(csv_file1, csv_file2, output_file):
    """ Merge the CSV files into a single file """
    df1 = pd.read_csv(csv_file1)
    print(f"df1 contains {len(df1)} rows.")
    df2 = pd.read_csv(csv_file2)
    print(f"df2 contains {len(df2)} rows.")
    # concat without headers (title names):
    df = pd.concat([df1, df2], ignore_index=True)
    # shuffle the rows
    df = df.sample(frac=1).reset_index(drop=True)
    # add headers:
    df.columns = ["image_path", "class_name", "description"]
    df.to_csv(output_file, index=False)
    print(f"df was saved to file {output_file}, contains {len(df)} rows.")


# main
if __name__ == '__main__':
    csv_file1 = "files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v2_300c.csv"
    csv_file2 = "files/clip_training_data/ImageNet1k/oxford_style/train.csv"
    output_file = "files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v2_300c_plus_imagenet1K_no_dogs.csv"

    csv_file1 = "files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v2_300c.csv"
    csv_file2 = "files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v3_300c.csv"
    output_file = "files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v2v3_600c.csv"

    csv_file1 = "files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v2v3_600c.csv"
    csv_file2 = "files/clip_training_data/ImageNet1k/oxford_style/train.csv"
    output_file = "files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v2v3_600c_plus_imagenet1K_no_dogs.csv"

    merge_clip_training_files(csv_file1, csv_file2, output_file)
