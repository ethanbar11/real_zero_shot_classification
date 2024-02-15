import argparse

import json

first_path = r'files/classnames/ImageNet1k/only_first_400.txt'
second_path = r'files/classnames/Dogs120/dogs120.txt'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class-path",
        type=str,
    )
    parser.add_argument(
        "--descriptions-file",
        type=str,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    forbidden_classes = []
    with open(args.class_path, 'r') as f:
        for line in f:
            forbidden_classes.append(line.strip().lower())
            forbidden_classes.append(line.strip().lower().replace('_', ' '))
            forbidden_classes.append(line.strip().lower().replace('-', ' '))


    with open(args.descriptions_file, 'r') as f:
        descriptions = json.load(f)

    final_descriptions = {}

    for cls_name, description in descriptions.items():
        parsed_cls_name = cls_name.replace('-', ' ').replace('_', ' ').replace('-', ' ').lower()
        valid = True
        for forbidden_cls in forbidden_classes:
            clean_forbidden_cls = forbidden_cls.replace('-', ' ').replace('_', ' ').replace('-', ' ').lower()
            if parsed_cls_name in forbidden_cls or forbidden_cls in parsed_cls_name:
                valid = False
                print(f'Found imagenet class {cls_name} that is relevant for {forbidden_cls}')
                break
        if valid:
            final_descriptions[cls_name] = description
    print(
        f'The final amount of descriptions is {len(final_descriptions)}, hich means we removed {len(descriptions) - len(final_descriptions)} descriptions')
    dataset_name = args.class_path.split('/')[-1].split('.')[0]
    output_path = args.descriptions_file.replace('.json', f'_filtered_{dataset_name}.json')
    print(f'Saving the descriptions to {output_path}')
    with open(output_path, 'w') as f:
        json.dump(final_descriptions, f, indent=4)
    classes = final_descriptions.keys()
    classes_path = output_path.replace('.json', f'_classes.txt')
    print(f'Saving the classes to {classes_path}')
    with open(classes_path, 'w') as f:
        for cls in classes:
            f.write(cls + '\n')
