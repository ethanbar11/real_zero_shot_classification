# pip install diffusers
import os
from diffusers import StableDiffusionPipeline
import torch
import time
import argparse
import json

def load_stable_diffusion_model(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

def generate_images(prompt, pipe):
    image = pipe(prompt).images[0]
    return image


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images from text using OpenAI\'s CLIP model')
    parser.add_argument(
        '-d', '--description_path',
        type=str,
        default='files/descriptors/CUB/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4.json',
        help='A json file with descriptions for each category'
    )
    parser.add_argument(
        '-o', '--output_path',
        type=str,
        default='./out/{}_{}',
        help='Path to save generated image to'
    )
    parser.add_argument(
        '-m', '--model_id',
        type=str,
        default='/shared-data5/guy/modelzoo/text2imaopenjourney',
        help='Model to use for text2image generation'
    )
    parser.add_argument(
        '-n', '--num_images',
        type=int,
        default=1,
        help='Number of images to generate'
    )
    parser.add_argument(
        "-l", "--limit",
        type=str,
        default=None
    )

    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### setup:
    limit = args.limit
    model_id = args.model_id
    output_folder = (args.output_path).format(
        os.path.basename(model_id).replace("-", "").replace(".", "_"),
        os.path.splitext(os.path.basename(args.description_path))[0]
    )
    os.makedirs(output_folder, exist_ok=True)

    #### load model:
    #model_id = '/shared-data5/guy/modelzoo/text2imaopenjourney'
    pipe = load_stable_diffusion_model(args.model_id, device)

    #### prepare descriptions:
    with open(args.description_path) as f:
        descriptions = json.load(f)
    # if limit is a file:
    if limit is not None and os.path.isfile(limit):
        # read lines, remove \n:
        with open(limit) as f:
            classes = f.readlines()
        classes = [c.strip() for c in classes]
        # filter descriptions:
        descriptions = {k: v for k, v in descriptions.items() if k in classes}
        descriptions = list(descriptions.items())
    elif limit is not None and limit.isdigit():
        descriptions = list(descriptions.items())[:int(limit)]
    else:
        descriptions = list(descriptions.items())

    #### generate images:
    gen_image_paths = []
    for category, description in descriptions:
        # create sub-folder:
        sub_folder = os.path.join(output_folder, "images", category.replace(" ", "_"))
        os.makedirs(sub_folder, exist_ok=True)
        # concat list of strings into one string:
        description = "\n".join(description)
        # save description to a file:
        with open(f"{sub_folder}/description.txt", 'w') as f:
            f.write(description)
        # generate synthetic images:
        for jj in range(args.num_images):
            out_path = f"{sub_folder}/image_{jj}.png"
            if os.path.exists(out_path):
                print(f'Image {out_path} already exists, skipping..')
                continue
            start = time.time()
            image = pipe(description).images[0]
            end = time.time()
            print(f'Generated image in {end - start:.2f} seconds..')
            image.save(out_path)
            print(f'Saved image to {out_path}..')
        # create a list of image paths:
        gen_image_paths.extend([os.path.join(sub_folder, f"image_{jj}.png") for jj in range(args.num_images)])
    # Removing the base path from each image path
    to_remove = os.path.join(output_folder, "images/")
    gen_image_paths = [path.replace(to_remove, '') for path in gen_image_paths]
    # Save the image paths with an index in a text file
    with open(os.path.join(output_folder, "images.txt"), "w") as f:
        for idx, path in enumerate(gen_image_paths, start=1):
            f.write(f"{idx} {path}\n")

####
    # generate images:

    # Henslow Sparrow:
    # prompt = (" is a small, brown songbird with a streaky breast and a white eye stripe. "
    #           "It has a short tail and a stout bill. The male has a distinctive black patch on its crown.")
    # out_path = "./henslow_sparrow.png"

    #  American goldfinch :
    # prompt = ("A small songbird with a distinctive black and yellow plumage. "
    #           "Adult males in the breeding season have bright yellow bodies, black caps, and white wingbars. "
    #           "Females and winter birds are more dull, with olive-green bodies and yellow heads. "
    #           "Both sexes have a pointed, notched tail and a conical bill.")
    # out_path = "./american_goldfinch.png"
