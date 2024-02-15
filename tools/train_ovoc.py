import datetime
import os
import sys
import json
import torch
import numpy as np
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ovod.utils.parser import parse_args, load_config
import ovod.utils.logger as logging
from ovod.utils.vis_optimization import create_search_graph_plot
from ovod.config.defaults import assert_and_infer_cfg
from ovod.datasets.build import build_dataset
from ovod.datasets.loader import construct_loaders
from ovod.classifiers.build import build_classifier
from ovod.llm_wrappers.build import build_llm_wrapper
from ovod.prompters.build import build_prompter
from ovod.optimizers.build import build_program_optimizer
from ovod.utils.class_name_fix import fix_classname_special_chars

logger = logging.get_logger(__name__)


def build_openset_model(cfg):
    """
    Build the programs for the openset model.
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
    out_directory = os.path.join(cfg.OUTPUT_DIR, dataset_name, openset, arch, backbone)#, description_type)
    if arch != "BaseAttributeClassifier":
        program_type = cfg.OPENSET.PATH_TO_PROGRAMS_FOLDER.replace("/", "_")
        out_directory = os.path.join(out_directory, program_type)
    # Create a string containing the current Day, Month, Year, Hour, Minute, Second
    date_as_string = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    out_directory = os.path.join(out_directory, date_as_string)
    os.makedirs(out_directory, exist_ok=True)
    logger.info(f"Saving results to {out_directory}")
    # Setup logging format, set debug print format.
    logging.setup_logging(out_directory, set_debug=True)

    # Print config.
    logger.info("Build with config:")
    logger.info(cfg)

    # Create testing dataset and loader.
    validation_dataset = build_dataset(f"{dataset_name}_synthetic", cfg, subset)
    img_example = validation_dataset[0][0].to(device) if torch.is_tensor(validation_dataset[0][0]) else validation_dataset[0][0]

    # synthetic data loaders:
    _, _, test_loader = construct_loaders(trainset=None, valset=None, testset=validation_dataset, cfg=cfg)
    test_info = f"Testing model over {len(validation_dataset)} examples, from {subset} split in {dataset_name} dataset."
    logger.info(test_info)

    # Build the prompter for generating programs:
    prompter = build_prompter(prompter_name=cfg.OPENSET.PROGRAM_PROMPT_TYPE,
                              base_folder=cfg.OPENSET.PROGRAM_PROMPT_FOLDER)

    # Build the LLM model for generating programs:
    llm_wrapper = build_llm_wrapper(cfg)

    # Build the classifier model and print model statistics.
    classifier = build_classifier(cfg, device=device)

    # descriptions setup:
    classnames_path = cfg.DATA.PATH_TO_CLASSNAMES
    with open(cfg.OPENSET.ATTRIBUTES_FILE) as f:
        descriptions = json.load(f)
    with open(classnames_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    # filter descriptions:
    descriptions = {k: v for k, v in descriptions.items() if k in classes}
    # Fix classes-names in descriptions
    descriptions = {fix_classname_special_chars(k): v for k, v in descriptions.items()}

    descriptions = list(descriptions.items())

    # Build the program optimizer:
    program_optimizer = build_program_optimizer(optimizer_name=cfg.OPENSET.PROGRAM_OPTIMIZER,
                                                cfg=cfg,
                                                logger=logger,
                                                classifier=classifier,
                                                descriptions=descriptions,
                                                prompter=prompter,
                                                llm_wrapper=llm_wrapper,
                                                out_directory=os.path.join(out_directory, "programs"))

    ### Optimization LOOP ###
    #########################
    total_results = []
    programs_folder = program_optimizer.init_search(init_programs_folder=cfg.OPENSET.PATH_TO_PROGRAMS_FOLDER,
                                                    test_info=test_info)

    for refinement_step in range(cfg.TRAIN.EPOCHS):

        ########## EVALUATE (FORWARD) PASS ##########

        #### EXECUTION
        test_outputs = program_optimizer.execution(device, test_loader, epoch=refinement_step)
        #        current_step_results['test_outputs'] = test_outputs
        logger.info(f"Done executing programs set {refinement_step} on few examples")
        #### EVALUATION
        confusion_matrix, _ = program_optimizer.evaluation(programs_folder, test_outputs)
        #        current_step_results['confusion_matrix'] = confusion_matrix
        logger.info(f"Done evaluating programs set {refinement_step} on few examples")
        #### SELECTION:
        to_keep, to_refine = program_optimizer.selection(confusion_matrix)
        logger.info(f"Done selecting programs from set {refinement_step} for next iteration")
        logger.info(f"Keeping {len(to_keep)} programs from set {refinement_step} for next iteration")

        # record:
        current_step_results = {
            'programs': read_programs(programs_folder), 'descriptions': descriptions, 'test_outputs': test_outputs,
            'confusion_matrix': confusion_matrix, 'to_keep': to_keep, 'to_refine': to_refine
        }
        # stop if to_remove becomes empty
        if to_refine == 0:
            break

        ########## UPDATE (BACKWARD) PASS ##########

        #### GENERATION:
        total_results.append(current_step_results)
        updated_programs_folder = program_optimizer.update(refinement_step, programs_folder, to_keep, to_refine,
                                                           context=total_results, image_example=img_example)
        # update programs_folder:
        programs_folder = updated_programs_folder
        # plot the search graph so far:
        create_search_graph_plot(os.path.join(out_directory, "programs"))
    logger.info("done. Final program version were stored in: %s", programs_folder)


def read_programs(programs_folder):
    programs_file_names = os.listdir(programs_folder)
    programs_file_names = [os.path.join(programs_folder, program) for program in programs_file_names]
    # remove __init__.py:
    programs_file_names = [program for program in programs_file_names if ".py" in program]
    programs = {}
    for program_file_name in programs_file_names:
        category = os.path.splitext(os.path.basename(program_file_name))[0]
        with open(program_file_name, 'r') as f:
            program = f.read()
        programs[category] = program
    return programs


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    build_openset_model(cfg)

####
# python tools/test_ovoc.py --cfg files/configs/CUB/CUB_BaseProgramClassifier_L14_TINY.yaml DATA.PATH_TO_CLASSNAMES files/classnames/CUB/opensets/tiny5/unseen.txt OPENSET.PATH_TO_PROGRAMS_FOLDER files/programs/CUB/gpt4_prompt_program_guybase/ DATA.ROOT_DIR out/openjourney_cub_gpt3_text-davinci-003_descriptions_ox_noname/ TEST.NUM_DEMO_PLOTS 0 OUTPUT_DIR tmp/
