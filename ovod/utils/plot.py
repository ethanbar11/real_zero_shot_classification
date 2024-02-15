import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread

def plot_image_and_progress_bar(image, labels, probabilities, output_path, gt=None):
  """
  Plot an image and a progress bar of labels with their corresponding probabilities.

  Args:
    image: A path to the image file.
    labels: A list of strings representing the labels.
    probabilities: A list of floats representing the probabilities of each label.
    output_path: The path to save the figure to.
    gt: The ground truth label of the image (optional).
  """

  #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))

  # Plot the image in the first panel. Fix image dimentions to RGB, if needed:
  if len(imread(image).shape) == 2:
      ax1.imshow(imread(image), cmap='gray')
  else:
      ax1.imshow(imread(image))
  ax1.axis('off')

  # Set the title of the first panel to the ground truth label if it is not None.
  if gt is not None:
    ax1.set_title(f"Ground truth label: {gt}")

  # Create a new image that is a copy of the original image.
  new_image = np.copy(imread(image))

  # Draw a white rectangle on the new image, with the same dimensions as the progress bar.
  ax2.add_artist(plt.Rectangle((0, 0), 1, len(labels), color='white'))

  # Plot the progress bar on the new image, over the white rectangle. add value to the right of the bar:
  for i, (label, probability) in enumerate(zip(labels, probabilities[0])):
      ax2.barh(i, probability, color='blue')
      ax2.system_text(probability, i, f"{probability:.3f}", ha='left', va='center')
      #ax2.text(probability, i, f"{probability:.2f}", ha='right', va='center')
      #ax2.text(probability, i, f"{probability:.2f}", ha='center', va='center')


  #ax2.barh(np.arange(len(labels)), probabilities[0], color='blue')


  ax2.set_yticks(np.arange(len(labels)))
  ax2.set_yticklabels(labels)
  ax2.set_xlim(0, 1)
  ax2.set_xlabel("Probability")

  # Save the new image.
  plt.savefig(output_path, bbox_inches='tight')
  plt.close(fig)



def plot_image_and_progress_bar_v2(image, labels, probabilities, gt, output_path):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
  ax1.imshow(mpimg.imread(image))
  # add title to ax1 panel with gt label:
  if gt is not None:
      ax1.set_title(f"Ground truth label: {gt}")

  ax1.axis('off')
  ax2.barh(np.arange(len(labels)), probabilities[0], color='blue')
  ax2.set_yticks(np.arange(len(labels)))
  ax2.set_yticklabels(labels)
  ax2.set_xlim(0, 1)
  ax2.set_xlabel("Probability")
  fig.savefig(output_path, bbox_inches='tight')
  plt.close(fig)

  def plot_image_prompt_response(image, prompt, response, fig, save_path=None, gt=None):
    """Plots an image, prompt, and response in a single JPG figure.

    Args:
      image: A numpy array representing the image.
      prompt: The prompt text.
      response: The response text.
      figsize: The size of the figure in inches.
      save_path: The path to save the figure to, or None to display the figure
        interactively.
    """

    # Add a subplot for the image.
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title('Image')

    # Add a subplot for the question and answer.
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.system_text(0.5, 0.8, f"Instruction: {prompt}", ha='center', va='center')
    ax2.system_text(0.5, 0.5, f"Response: {response}", ha='center', va='center')
    if gt is not None:
      ax2.system_text(0.5, 0.3, f"GT: {gt}", ha='center', va='center')
    # Break the answer into lines that fit in the second panel.
    # answer_lines = wrap_text(response, fig.get_size_inches()[1] * 0.5)
    # for i in range(len(answer_lines)):
    #     ax2.text(0.5, 0.3 - i / (len(answer_lines) - 1), answer_lines[i], ha='center', va='center')
    ax2.set_title('Question and Answer')
    ax2.axis('off')
    fig.tight_layout()

    if save_path is not None:
      fig.savefig(save_path)
    else:
      plt.show()


def wrap_text(text, width):
  """Wraps the text to fit within the specified width.

  Args:
    text: The text to wrap.
    width: The width of the text box.

  Returns:
    A list of strings, where each string is a line of text that fits within the specified width.
  """

  lines = []
  current_line = ''
  for word in text.split():
    if len(current_line + word) > width:
      lines.append(current_line)
      current_line = word
    else:
      current_line += ' ' + word

  if current_line:
    lines.append(current_line)

  return lines

def plot_image_prompt_response(image, prompt, response, fig, save_path=None, gt=None):
    """ Plots an image, prompt, and response in a single JPG figure.

    Args:
      image: A numpy array representing the image.
      prompt: The prompt text.
      response: The response text.
      figsize: The size of the figure in inches.
      save_path: The path to save the figure to, or None to display the figure
        interactively.
    """

    # Add a subplot for the image.
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(image)
    ax1.set_title('Image')

    # Add a subplot for the question and answer.
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.system_text(0.5, 0.8, f"Instruction: {prompt}", ha='center', va='center')
    # Option 1: normal text:
    ax2.system_text(0.5, 0.5, f"Response: {response}", ha='center', va='center')
    # Option 2: Break the answer into lines that fit in the second panel.
    # answer_lines = wrap_text(response, fig.get_size_inches()[1] * 0.5)
    # for i in range(len(answer_lines)):
    #     ax2.text(0.5, 0.5 - i * 0.1, answer_lines[i], ha='center', va='center')

    if gt is not None:
      ax2.system_text(0.5, 0.2, f"GT: {gt}", ha='center', va='center')
    ax2.set_title('Question and Answer')
    ax2.axis('off')
    fig.tight_layout()

    if save_path is not None:
      fig.savefig(save_path)
    else:
      plt.show()


def wrap_text(text, width):
    """Wraps the text to fit within the specified width.

    Args:
      text: The text to wrap.
      width: The width of the text box.

    Returns:
      A list of strings, where each string is a line of text that fits within the specified width.
    """

    lines = []
    current_line = ''
    for word in text.split():
      if len(current_line + word) > width:
        lines.append(current_line)
        current_line = word
      else:
        current_line += ' ' + word

    if current_line:
      lines.append(current_line)

    return lines
