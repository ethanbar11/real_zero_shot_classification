import torch
from PIL import Image
import open_clip
from typing import List
from ovod.utils.plot import plot_image_and_progress_bar, plot_image_and_progress_bar_v2

class OpenClipModel:
    def __init__(self, model_name, model_path):
        self.load_model(model_name, model_path)
        self.set_prompt("")

    def set_prompt(self, template):
        self.template = template
        print(f"template: {self.template}")


    def load_model(self, model_name, model_path):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=model_path)
        self.train_preprocess = self.preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)    # ('ViT-B-32')

    def process_image(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        return image

    def process_text(self, text):
        text = [self.template + l for l in text]
        text = self.tokenizer(text)
        return text

    def run_itm(self, image, text):
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            #
            #text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            text_probs = image_features @ text_features.T
        return text_probs

    def plot_itm_results(self, image: str, labels: List, probs: List, gt: str = None, output_path: str = None):
        """ Create a figure, with two panels, in which the image is in one panel, and a progress bas of labels with their
            corresponding probabilities is in another panel. Remove axis on the image, and save to output_path.
        """
        plot_image_and_progress_bar(image, labels, probs, output_path, gt)



# main:
if __name__ == '__main__':
    model_path = "/shared-data5/guy/modelzoo/openclip/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin"
    model_name = 'ViT-B-32'
    model = OpenClipModel(model_name, model_path)

    image_path = "samples/dog-png-12.png"
    labels = ["a diagram", "a dog", "a cat"]

    image = model.process_image(image_path)
    text = model.process_text(labels)
    text_probs = model.run_itm(image, text)
    # print all labels with their corresponding probabilities
    for i, label in enumerate(labels):
        print(f"{label}: {text_probs[0][i]:.5f}")
    # plot image and labels with their corresponding probabilities
    output_path = "dog-png-12_itm.png"
    model.plot_itm_results(image_path, labels, text_probs, output_path)
    print(f"image-text matching on {image_path} saved to {output_path}")


