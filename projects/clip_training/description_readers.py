import yaml


def yaml_description_reader(attributes_as_txt):
    finished = False
    while not finished:
        try:
            atts_in_original_list = yaml.safe_load(attributes_as_txt)
            finished = True
        except Exception as e:
            attributes_as_txt = attributes_as_txt[:e.problem_mark.index] + ' , ' + attributes_as_txt[
                                                                                   e.problem_mark.index + 1:]
            finished = False

    atts_to_add = []
    for att in atts_in_original_list:
        if att['is_relevant']:
            atts_to_add.append(att['sentence'])
    return atts_to_add

def list_description_reader(attributes_as_txt):
    # They should be already as a list
    return attributes_as_txt
def build_description_reader(description_reader_type):
    if description_reader_type=='yaml':
        return yaml_description_reader
    elif description_reader_type=='list':
        return list_description_reader