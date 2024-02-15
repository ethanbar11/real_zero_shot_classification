import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def plot_attributes_predictions(img_path: str, output: dict, out_directory: str):
    """
    plot a list of attributes and their scores on the right side of the image, and save to out_directory
    """
    label_pred = output["label_pred"]
    attrs_pred_names = output["attrs_pred"]
    attrs_pred_scores = output["explanation_pred"]
    label_gt = output["label_gt"]
    attrs_gt_names = output["attrs_gt"]
    attrs_gt_scores = output["explanation_gt"]

    # Create a large figure and add a subplot for the image, then a subplot for bar chart of the scores and attributes:
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(211)
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"pred: {label_pred}, gt: {label_gt}")

    # Create a new subplot for the attributes and scores as horizontal bar chart to the right of the image.
    # Another value is added for each column, which is the value for attrs_gt_names and attrs_gt_scores:
    ax2 = fig.add_subplot(212)
    ax2.set_yticks(np.arange(len(attrs_gt_names) + len(attrs_pred_names)))
    ax2.set_yticklabels(attrs_gt_names + attrs_pred_names, rotation=0)
    ax2.set_xlim([0.1, 0.5])
    ax2.set_xticks(np.arange(0.1, 0.5, 0.1))
    ax2.grid(True)

    # Add a bar chart of the scores
    plt.barh(attrs_pred_names, attrs_pred_scores[:len(attrs_pred_names)])
    plt.xlabel("Scores")
    plt.ylabel("Attributes")
    plt.title("Attribute scores")

    # Add another bar chart for ground truth scores
    plt.barh(attrs_gt_names, attrs_gt_scores[:len(attrs_gt_names)], color='red')
    plt.legend(['Predicted', 'Ground Truth'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_directory, os.path.basename(img_path)))
    plt.close(fig)


import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import os

def plot_vqa_prediction(img_path, output, out_directory):
    """
    Plot a visual question answering result with the image, question, and answer in separate panes,
    and save the plot to the specified output directory. The image pane is larger than the text panes.
    """
    label_pred = output["label_pred"]
    label_gt = output["label_gt"]
    question = output["question"]
    answer = output["answer"]

    # Create a figure with a gridspec layout
    fig = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])  # Width ratio for image is larger

    # Subplot for the image
    ax_img = plt.subplot(gs[0])
    img = Image.open(img_path)
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title(f"pred: {label_pred}, gt: {label_gt}")

    # Subplot for the question
    ax_question = plt.subplot(gs[1])
    ax_question.axis('off')
    ax_question.text(0.1, 0.5, f"Question: {question}", fontsize=12, wrap=True, verticalalignment='center')

    # Subplot for the answer
    ax_answer = plt.subplot(gs[2])
    ax_answer.axis('off')
    ax_answer.text(0.1, 0.5, f"Answer: {answer}", fontsize=12, wrap=True, verticalalignment='center')

    # Save the plot to the specified output directory
    plt.tight_layout()
    plt.savefig(os.path.join(out_directory, os.path.basename(img_path)))
    plt.close(fig)



def plot_vqa_prediction_to_screen(img_path, output):
    """
    Plot a visual question answering result on the right side of the image and display it on the screen.
    """
    label_pred = output["label_pred"]
    label_gt = output["label_gt"]
    question = output["question"]
    answer = output["answer"]

    # Create a large figure and add a subplot for the image:
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)  # Change to 121 for two subplots side by side
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"pred: {label_pred}, gt: {label_gt}")

    # Create a new subplot for the question and answer:
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    ax2.text(0.1, 0.8, f"Question: {question}", fontsize=12, wrap=True)
    ax2.text(0.1, 0.5, f"Answer: {answer}", fontsize=12, wrap=True)

    # Display the plot on the screen:
    plt.tight_layout()
    plt.show()
