import pickle
import io
import random

from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter
import textwrap


def plot_pickle(pickle_file_path: str):
    # open outputs.pickle and plot results:
    with open(pickle_file_path, 'rb') as f:
        outputs = pickle.load(f)
    # outputs is a list of dictionaries, each dictionary is a result of a single image
    print(f'Done computing object programs for {len(outputs)} images')


def draw_text(draw, position, text, font, max_width):
    lines = textwrap.wrap(text, width=max_width)
    y = position[1]
    for line in lines:
        draw.system_text((position[0], y), line, font=font, fill="black")
        y += font.getsize(line)[1]


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def plot_program_with_bounding_boxes(name, attributes, score, func_text, image, output_path):
    try:
        font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        font = ImageFont.load_default()

    image_width, image_height = image.size
    extended_height = image_height + 100  # Add 100 pixels to the height for the text area
    extended_width = image_width + 200  # Add 200 pixels to the width for the side text
    new_image = Image.new("RGB", (extended_width, extended_height), "white")
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)
    i = 0
    for name, part in attributes.items():
        box = part['bbox']
        box_color = "black"
        text_color = box_color
        number = str(i + 1)
        new_box = ((box[0][0], image_height - box[0][1]), (box[1][0], image_height - box[1][1]))
        draw.rectangle(new_box, outline=box_color, width=2)
        draw.text((box[0][0] - 5, new_box[0][1] + 5), number, font=font, fill=text_color)
        description = str(part)
        draw_text(draw, (20, image_height + 20 * i), description, font=font, max_width=image_width)
        i += 1

    draw.text((image_width + 20, 0), func_text + f"\nName: {name}\nOutput Score: {score}", font=font,
              fill="black")
    new_image.save(output_path)


def draw_bbox_on_image(image, attributes):
    try:
        font = ImageFont.truetype("HelveticaNeueLight.ttf", size=30)
    except IOError:
        font = ImageFont.load_default()

    image_width, image_height = image.size
    draw = ImageDraw.Draw(image)
    for name, part in attributes.items():
        if type(part) == dict and 'bbox' in part:
            box = part['bbox']
            box_color = "black"
            text_color = "blue"
            new_box = ((box[0][0], image_height - box[0][1]), (box[1][0], image_height - box[1][1]))
            draw.rectangle(new_box, outline=box_color, width=2)
            draw.text((box[0][0] - 5, new_box[0][1] + 5), name, font=font, fill=text_color)


def plot_program(program: str, program_output_dict: dict, program_name: str, program_score: float, image: Image,
                 output_file: str, with_bounding_box=False, image_patch=None):
    """
    Display the given Python program alongside the provided image.

    :param program: The Python code as a string.
    :param program_output_dict: A dictionary containing program outputs.
    :param image: The PIL.Image object on which the program was executed.
    """

    # Format the outputs and append them to the program text
    outputs = "#Attribute Scores Output:\n"
    if program_output_dict is not None:
        for key1, subdict in program_output_dict.items():
            try:
                if type(subdict) == dict:
                    for key2, value in subdict.items():
                        outputs += f"attributes['{key1}']['{key2}'] = {value}\n"
                else:
                    outputs += f"attributes['{key1}'] ={subdict}\n"
            except Exception as e:
                print(f'Exception in visualization  {program_name}: {e}. continuing')

    exists_outputs = "#Exists Results:\n"
    if image_patch and image_patch.exists_full_results:
        for key, result in image_patch.exists_full_results['exists_results'].items():
            exists_outputs += f"exists_results['{key}'] = {result}\n"

    full_text = f" # Visual program for {program_name} \n" + program + "\n\n" \
                + exists_outputs + "\n" + outputs + "\n Total program score: " + str(program_score)

    # Use pygments to generate an image with highlighted code
    highlighted_code = highlight(full_text, PythonLexer(), ImageFormatter(line_numbers=True))

    # Convert the highlighted code bytes to a PIL Image
    code_image = Image.open(io.BytesIO(highlighted_code))

    # Determine the size for the final combined image
    total_width = image.width + code_image.width
    height = max(image.height, code_image.height)

    # Create an empty image with the determined size
    combined = Image.new('RGB', (total_width, height), color=(255, 255, 255))

    if with_bounding_box:
        draw_bbox_on_image(image, program_output_dict)
    # Paste the input image and the code image onto the combined image

    combined.paste(image, (0, 0))
    combined.paste(code_image, (image.width, 0))

    # Show the final combined image
    # combined.show()
    combined.save(output_file)
    return output_file


def integrate_outputs(program: str, program_output_dict: dict):
    # Split the program into lines
    lines = program.split("\n")

    # Find where to insert the outputs
    for i, line in enumerate(lines):
        if "attributes['name1']" in line:
            for key, value in program_output_dict['name1'].items():
                lines.insert(i + 1, f"# Output: attributes['name1']['{key}'] = {value}")
                i += 1  # Adjust the index since we added a line
        # Similarly, add conditions for other keys as necessary

    # Join the modified lines back
    return "\n".join(lines)


def plot_program_integrated_output(program: str, program_output_dict: dict, image: Image, output_file: str):
    """
    Display the given Python program alongside the provided image with integrated outputs.

    :param program: The Python code as a string.
    :param program_output_dict: A dictionary containing program outputs.
    :param image: The PIL.Image object on which the program was executed.
    """

    full_text = integrate_outputs(program, program_output_dict)

    # Use pygments to generate an image with highlighted code
    highlighted_code = highlight(full_text, PythonLexer(), ImageFormatter(line_numbers=True))

    # Convert the highlighted code bytes to a PIL Image
    code_image = Image.open(io.BytesIO(highlighted_code))

    # Determine the size for the final combined image
    total_width = image.width + code_image.width
    height = max(image.height, code_image.height)

    # Create an empty image with the determined size
    combined = Image.new('RGB', (total_width, height), color=(255, 255, 255))

    # Paste the input image and the code image onto the combined image
    combined.paste(image, (0, 0))
    combined.paste(code_image, (image.width, 0))

    # Show the final combined image
    combined.save(output_file)


# main code:
if __name__ == '__main__':
    plot_pickle(r'/home/ethan/phrase_grounding/viper/outputs.pickle')

    # Test
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    program_str = """
    def hello():
        print('Hello, World!')
        attributes['name1']['name1_a']
    """
    outputs = {
        'name1': {
            'name1_a': 5.123,
            'name1_b': 10.456
        },
        'name2': {
            'name2_a': 20.789
        }
    }
    # plot_programs1(program_str, {}, img)
    plot_program(program_str, outputs, img, "stam.png")
