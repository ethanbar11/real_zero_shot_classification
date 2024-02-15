"""Configs."""
from fvcore.common.config import CfgNode
from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "CUB"

# Method.
_C.TRAIN.METHOD = "vanilla"

# Input size at training
_C.TRAIN.INPUT_SIZE = 256

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 8

# Number of epochs.
_C.TRAIN.EPOCHS = 20

_C.SAVE_RESULTS_EVERY= 5

# Number of epochs.
_C.TRAIN.BREADTH = 4

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# option to overfit (debug mode)
_C.DATA_LOADER.OVERFIT = False

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "BaseAttributeClassifier"

# Model name
_C.MODEL.BACKBONE = "/shared-data5/guy/modelzoo/clip/ViT-B-16.pt"
_C.MODEL.MODEL_SIZE = "ViT-B-16"
_C.EXAMPLE_DIRECTORY = "awefawef"

_C.MODEL.NUM_CLASS = 2 #400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# weights to init the model
_C.MODEL.WEIGHTS = None

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False

# If True, detach the final fc layer from the network, by doing so, only the
# final fc layer will be trained.
_C.MODEL.DETACH_FINAL_FC = False

# If True, frozen batch norm stats during training.
_C.MODEL.FROZEN_BN = False

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# data root (e.g., for SUN)
_C.DATA.ROOT_DIR = None

# path to info file
_C.DATA.PATH_TO_INFO_FILE = None

# path to semantics file, e.g., attributes file
_C.DATA.PATH_TO_SEMANTICS_FILE = None

# text file containing all paths (relative ro data_root) to valid data items:
_C.DATA.PATH_TO_CLASSNAMES = None

# Path to filenames list
_C.DATA.SPLIT_OF_FILENAMES = None

# percent of training data
_C.DATA.PERCENT_TRAIN = 0.8

# Apply transform normalization or not when fetching.
_C.DATA.APPLY_NORMALIZE_TRANSFORM = True

# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()

# Whether to enable randaug.
_C.AUG.ENABLE = False

# Augmentation probability (Portion of augmented images during training: [0,1])
_C.AUG.PROB = 0.75

# Augmentation type
_C.AUG.METHOD = "basic"

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# optimizer.
_C.SOLVER.OPTIMIZER = 'Adam'

# Base learning rate.
_C.SOLVER.BASE_LR = 0.0001

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False

# Config for layers' weights and activations visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS = CfgNode()

# If False, skip model visualization.
_C.TENSORBOARD.MODEL_VIS.ENABLE = False

# If False, skip visualizing model weights.
_C.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = False

# If False, skip visualizing model activations.
_C.TENSORBOARD.MODEL_VIS.ACTIVATIONS = False

# If False, skip visualizing input image.
_C.TENSORBOARD.MODEL_VIS.INPUT_IMAGE = False


# List of strings containing data about layer names and their indexing to
# visualize weights and activations for. The indexing is meant for
# choosing a subset of activations outputed by a layer for visualization.
# If indexing is not specified, visualize all activations outputed by the layer.
# For each string, layer name and indexing is separated by whitespaces.
# e.g.: [layer1 1,2;1,2, layer2, layer3 150,151;3,4]; this means for each array `arr`
# along the batch dimension in `layer1`, we take arr[[1, 2], [1, 2]]
_C.TENSORBOARD.MODEL_VIS.LAYER_LIST = []
# Top-k predictions to plot on videos
_C.TENSORBOARD.MODEL_VIS.TOPK_PREDS = 1
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.COLORMAP = "Pastel2"
# Config for visualization video inputs with Grad-CAM.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM = CfgNode()
# Whether to run visualization using Grad-CAM technique.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
# CNN layers to use for Grad-CAM. The number of layers must be equal to
# number of pathway(s).
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = []
# If True, visualize Grad-CAM using true labels for each instances.
# If False, use the highest predicted class.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP = "viridis"

# Config for visualization for wrong prediction visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.WRONG_PRED_VIS = CfgNode()
_C.TENSORBOARD.WRONG_PRED_VIS.ENABLE = False
# Folder tag to origanize model eval images under.
_C.TENSORBOARD.WRONG_PRED_VIS.TAG = "Incorrectly classified images."
# Subset of labels to visualize. Only wrong predictions with true labels
# within this subset is visualized.
_C.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH = ""

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "CUB"

# Dataset split for testing.
_C.TEST.SUBSET = "test"

# Number of images to test, set -1 to test all images.
_C.TEST.SAMPLE_ITEMS = -1

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Test tasks to preform, assuming several heads in DNN and several evaluation tasks are needed.
_C.TEST.TASKS = ["similarity", "sharpness"]

# Number of random test examples to plot and visualize results
_C.TEST.NUM_DEMO_PLOTS = 10

# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""

# ---------------------------------------------------------------------------- #
# Demo options
# ---------------------------------------------------------------------------- #
_C.DEMO = CfgNode()

# Run model in DEMO mode.
_C.DEMO.ENABLE = False

# Custom width for reading input video data.
_C.DEMO.DISPLAY_WIDTH = 0
# Custom height for reading input video data.
_C.DEMO.DISPLAY_HEIGHT = 0
# Path to Detectron2 object detection model configuration,
# If specified, the visualized outputs will be written this a video file of
# this path. Otherwise, the visualized outputs will be displayed in a window.
_C.DEMO.OUTPUT_FILE = ""
# Frames per second rate for writing to output video file.
# If not set (-1), use fps rate from input file.
# Option: {"thres", "top-k"}
_C.DEMO.VIS_MODE = "thres"
# Threshold for common class names.

# ---------------------------------------------------------------------------- #
# ImageNet dataset options
# ---------------------------------------------------------------------------- #
_C.ImageNet = CfgNode()

# data root (e.g., for ImageNet)
_C.ImageNet.PATH_TO_DATA_DIR = "/shared-data5/guy/data/ILSVRC2012/validation/"

# text file containing all paths (relative ro data_root) to valid data items:
_C.ImageNet.VALID_FILENAMES = "files/filenames/ImageNet_all.txt"

# Path to filenames list
_C.ImageNet.SPLIT_OF_FILENAMES = "files/filenames/imagenet_split2/"

# data type, e.g., "natural" or "uls"
_C.ImageNet.DATA_TYPE = "natural"

# ---------------------------------------------------------------------------- #
# AK153 dataset options
# ---------------------------------------------------------------------------- #
_C.AK153 = CfgNode()

# data root (e.g., for ImageNet)
_C.AK153.PATH_TO_DATA_DIR = "/shared-data5/guy/data/SRI_AK153/"

# text file containing all paths (relative ro data_root) to valid data items:
_C.AK153.VALID_FILENAMES = "files/filenames/AK153_all.txt"

# Path to filenames list
_C.AK153.SPLIT_OF_FILENAMES = "files/filenames/AK153_split/"

# data type, e.g., "natural" or "uls"
_C.AK153.DATA_TYPE = "uls"

# ---------------------------------------------------------------------------- #
# Show data options
# ---------------------------------------------------------------------------- #
_C.SHOWDATA = CfgNode()

# Run Show Data mode, namely going over few data items from dataset.
_C.SHOWDATA.ENABLE = False

# In Show Data mode, plotting data items∂.
_C.SHOWDATA.PLOT = True

# In Show Data mode, check validity of data items.
_C.SHOWDATA.CHECK = False

# Show metric together with data items
_C.SHOWDATA.SHOW_METRIC = True

# Show dataset index starting at
_C.SHOWDATA.RANGE_START = 100

# Show dataset index ending at
_C.SHOWDATA.RANGE_END = 150

# Show dataset items at step size:
_C.SHOWDATA.STEP = 5

# ---------------------------------------------------------------------------- #
# OpenSet options.
# ---------------------------------------------------------------------------- #
_C.OPENSET = CfgNode()

_C.OPENSET.ATTRIBUTES_FILE = "files/descs/attributes/cub_gpt3_text-davinci-003_descriptions_ox_prompt.json"

_C.OPENSET.PATH_TO_PROGRAMS_FOLDER = "files/programs/set1"

_C.OPENSET.PATH_TO_SYNTHETIC_FOLDER = "/shared-data5/guy/exps/grounding_synthetic/openjourney_cub_gpt3_text-davinci-003_descriptions_ox_noname"

_C.OPENSET.PROGRAM_PROMPT_TYPE = "program_base"

_C.OPENSET.SYNTHETIC_IMAGE_AMOUNT = 5

_C.OPENSET.PROGRAM_PROMPT_FOLDER = "files/prompts/program_ethan1/"

_C.OPENSET.OPENAI_MODEL = 'gpt-3.5-turbo'

_C.OPENSET.PROGRAM_OPTIMIZER = 'chain_search'

_C.OPENSET.DETECTION_THRESHOLD = 0.17

_C.OPENSET.MAX_NUM_INTERVALS = 10

# ---------------------------------------------------------------------------- #
# LLM wrapper options.
# ---------------------------------------------------------------------------- #
_C.LLM_WRAPPER = CfgNode()

_C.LLM_WRAPPER.PARAMS = CfgNode()

_C.LLM_WRAPPER.ARCH = ''

_C.LLM_WRAPPER.PARAMS.model = ''

_C.LLM_WRAPPER.PARAMS.temperature = 1.0

_C.LLM_WRAPPER.PARAMS.top_p = 1.0

_C.LLM_WRAPPER.PARAMS.frequency_penalty = 0.0

_C.LLM_WRAPPER.PARAMS.presence_penalty = 0.0

_C.LLM_WRAPPER.PARAMS.stop = 'STOP'

_C.LLM_WRAPPER.PARAMS.load_in_8bit = False

_C.LLM_WRAPPER.PARAMS.max_new_tokens = None
_C.LLM_WRAPPER.PARAMS.min_new_tokens = None
_C.LLM_WRAPPER.PARAMS.device = 'auto'
_C.LLM_WRAPPER.PARAMS.repetition_penalty = 1.0
_C.LLM_WRAPPER.PARAMS.top_k = 1
_C.LLM_WRAPPER.PARAMS.do_sample = True
_C.MODEL.SERVER_URL= ''
_C.MODEL.SYSTEM_PROMPT_LOCATION= ''
_C.MODEL.TEMPERATURE= 0.0
_C.MODEL.TOP_P= 0.0
_C.MODEL.MAX_NEW_TOKENS= 0
_C.MODEL.MODEL_ID= 'WOHO'
_C.MODEL.QUESTIONS_PATH= ''



# ---------------------------------------------------------------------------- #
# MISC options
# ---------------------------------------------------------------------------- #
# Seed
_C.SEED = 2023
# Number of epochs for data loading benchmark.
_C.OUTPUT_DIR = "/shared-data5/guy/exps/grounding_ge"

# Add custom config with default values.
custom_config.add_custom_config(_C)


def assert_and_infer_cfg(cfg):

    # TRANSCRIPT assertions.
    assert cfg.AUG.METHOD in ["basic", "bla1", "bla2"]

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()


if __name__ == '__main__':
    cfg = get_cfg()
