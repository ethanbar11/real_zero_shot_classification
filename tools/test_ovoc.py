import os
import sys
import shutil
# import ipdb
import torch
import numpy as np
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ovod.utils.parser import parse_args, load_config
import ovod.utils.logger as logging
from ovod.utils.results_savers import build_results_saver
from ovod.config.defaults import assert_and_infer_cfg
from ovod.datasets.build import build_dataset
from ovod.datasets.loader import construct_loaders
from ovod.classifiers.build import build_classifier
from ovod.engine.loops import test

logger = logging.get_logger(__name__)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def test_openset(cfg):
    """
    Perform open-set testing on AttributeClassifier and ProgramClassifier model. Performs evaluation using `ClassEvaluator`
    Plot the results if needed
    Args:
        cfg (CfgNode): configs. Details can be found in
            config/defaults.py
    """
    # Set up environment.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seed from configs.
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    import random, os
    seed = cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Setup logging format.
    dataset_name, subset, arch = cfg.TEST.DATASET, cfg.TEST.SUBSET, cfg.MODEL.ARCH
    if cfg.MODEL.BACKBONE == "":
        backbone = ""
    else:
        backbone = (os.path.splitext(os.path.basename(cfg.MODEL.BACKBONE))[0]).replace('-', '_')
    description_type = os.path.splitext(os.path.basename(cfg.OPENSET.ATTRIBUTES_FILE))[0]
    description_type = description_type.split("descriptions_")[1]
    openset = cfg.DATA.PATH_TO_CLASSNAMES.split("/")[-2].replace("/", "")
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_directory = os.path.join(cfg.OUTPUT_DIR, dataset_name, openset, arch, backbone, description_type, time_str)
    if arch != "BaseAttributeClassifier":
        program_type = cfg.OPENSET.PATH_TO_PROGRAMS_FOLDER.replace("/", "_")
        out_directory = os.path.join(out_directory, program_type)
    os.makedirs(out_directory, exist_ok=True)
    logging.setup_logging(out_directory, set_debug=False)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_classifier(cfg, device=device)
    logger.info("Model: {}".format(model))

    # Create testing dataset and loader.
    test_dataset = build_dataset(dataset_name, cfg, subset)

    # data loaders:
    _, _, test_loader = construct_loaders(trainset=None, valset=None, testset=test_dataset, cfg=cfg)
    test_info = f"Testing model over {len(test_dataset)} examples, from {subset} split in {dataset_name} dataset."
    logger.info(test_info)

    # load loss, to compare with val loss (TODO later)
    # create a csv results file

    results_saver = build_results_saver(cfg, model, test_info)

    # Perform test on the entire dataset.
    criterion = None
    test(mode='test',
         epoch=1,
         loader=test_loader,
         model=model,
         device=device,
         criterion=criterion,
         cfg=cfg,
         prossesID=None,
         results_saver=results_saver)

    logger.info("Done computing attribute scores for test examples")
    results_saver.save()

    # evaluate the results
    results_saver.calc_metrics()
    results_saver.visualize()

    print("done.")


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    test_openset(cfg)
