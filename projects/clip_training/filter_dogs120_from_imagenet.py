first_path = r'files/classnames/ImageNet1k/only_first_400.txt'
second_path = r'../../files/classnames/Dogs120/dogs120.txt'

first_classes = []
with open(first_path, 'r') as f:
    for line in f:
        first_classes.append(line.strip().lower())

second_classes = []
with open(second_path, 'r') as f:
    for line in f:
        second_classes.append(line.strip().lower())

final_classes = []
for first_cls in first_classes:
    exists = False
    for second_cls in second_classes:
        if first_cls in second_cls or second_cls in first_cls:
            exists = True
            print(first_cls)
            break
    if not exists:
        final_classes.append(first_cls)
# Take all the classes that are not in the intersection
out_path = r'files/classnames/ImageNet1k/only_first_400_without_dogs120.txt'
with open(out_path,'w') as f:
    for cls in final_classes:
        f.write(cls+'\n')
