def fix_classname_special_chars(category_name):
    k = category_name.split('.')[-1].replace('_', ' ')
    split_key = k.split(' ')
    if len(split_key) > 2:
        k = '-'.join(split_key[:-1]) + " " + split_key[-1]
    return k
