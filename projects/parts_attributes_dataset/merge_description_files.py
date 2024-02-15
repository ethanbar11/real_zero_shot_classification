import glob
import json

def merge_description_files(json_files, output_file):
    """ Merge the Json files into a single file """
    all_items = dict()
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        print(f"data contains {len(data)} items.")
        #all_items += data
        # merge data dict with all_items dict:
        all_items = {**all_items, **data}

    # remove duplicated keys in the all_items dict:
    all_items_without_duplicates = dict()
    for k, v in all_items.items():
        if k not in all_items_without_duplicates:
            all_items_without_duplicates[k] = v
        else:
            print(f"key {k} already exists in all_items_without_duplicates dict.")

    all_items = all_items_without_duplicates


    print(f"all_items contains {len(all_items)} items.")

    # data = dict()
    # for k, v in all_items.items():
    #     if k not in data:
    #         data[k] = v
    #     else:
    #         print(f"key {k} already exists in data dict.")
    #
    #
    # # write to file
    with open(output_file, "w") as f:
        json.dump(all_items, f, indent=4)
    print(f"data was saved to file {output_file}, contains {len(all_items)} items.")


    # df1 = pd.read_csv(csv_file1)
    # print(f"df1 contains {len(df1)} rows.")
    # df2 = pd.read_csv(csv_file2)
    # print(f"df2 contains {len(df2)} rows.")
    # # concat without headers (title names):
    # df = pd.concat([df1, df2], ignore_index=True)
    # # shuffle the rows
    # df = df.sample(frac=1).reset_index(drop=True)
    # # add headers:
    # df.columns = ["image_path", "class_name", "description"]
    # df.to_csv(output_file, index=False)
    # print(f"df was saved to file {output_file}, contains {len(df)} rows.")


# main
if __name__ == '__main__':
    output_file = "files/descriptors/ImageNet21k/descriptions_ox_noname_imagenet21k_all.json"
    # read all jsons from folder:
    jsons = glob.glob("files/descriptors/ImageNet21k/descriptions_ox_noname_imagenet21k/*.json")
    # merge all jsons into a single file:
    merge_description_files(jsons, output_file)





