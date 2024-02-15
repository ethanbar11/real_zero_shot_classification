from .BasePrompter import BasePrompter
from ..utils.text_process import check_percent


def filter_name_from_description(description: str, category: str) -> str:
    """ filter the result to remove the category name from the description"""

    # consider cases where category name is shown in singular, following by A, An, The, etc.:
    description = description.replace(f"A {category}", "It").replace(f"An {category}", "It").replace(
        f"A {category.lower()}",
        "It").replace(
        f"An {category.lower()}", "It")
    description = description.replace(f"The {category}", "It").replace(f"The {category.lower()}", "It")
    description = description.replace(f"{category}", "It").replace(f"{category.lower()}", "It")
    # consider cases where category name words are separated by hyphen:
    description = description.replace(f"{category.replace(' ', '-')}", "It").replace(
        f"{category.lower().replace(' ', '-')}",
        "It")
    # consider cases where there are two words in category name, one starts with capital letter and the other start with lower letter, e.g., American eagle, american Eagle, etc.:
    description = description.replace(f"{category.title().replace(' ', '')}", "It").replace(
        f"{category.lower().title().replace(' ', '')}", "It")

    description = description.replace(f"{category.title()}", "It").replace(f"{category.lower().title()}", "It")
    # consider cases where category name is shown in plural, followed by s, es, or ies, etc.:
    description = description.replace(f"{category}s", "They").replace(f"{category}es", "They").replace(f"{category}ies",
                                                                                                       "They")
    forbidden_words = ["Albatross", "Auklet", "Widow", "Flycatcher", "Grebe", "Violetear", "Kingfisher", "Oriole",
                       "Sparrow", "Tanager", "Tern", "Vireo", "Warbler", "Wren", "Woodpecker", "Woodcreeper"]
    for word in forbidden_words:
        description = description.replace(word, 'bird')
    # consider cases where category name has several words, which may or may not appear with a hyphen or comma.
    # As example, Chuck-will Widow can appear as Chuck-will's Widow or Chuck-will's-widow.
    # In both cases we want to remove the category name from the description.
    # Method: if category name is long (in terms of chars), check if 90% of the name exists, and if so remove it from description:
    # if len(category) > 10:   # TODO
    #     if check_percent(category, description, percent=90):
    #         description = description.replace(category, "It")

    return description


class ObjectDescriptionPrompter(BasePrompter):
    def __init__(self, base_folder, include_object_name: bool = True):
        super().__init__(base_folder)  # Calling the BasePrompter's __init__ method
        self.include_object_name = include_object_name

    def get_prompt(self, text_query):
        raise NotImplementedError

    def include_object_name_in_prompt(self) -> bool:
        return self.include_object_name
