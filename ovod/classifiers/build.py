import torch
from fvcore.common.config import CfgNode

from ovod.classifiers.FineTunedAttributeClassifier import FineTunedAttributeClassifier
from ovod.classifiers.LLaVaGptClassifier import LLaVaGptClassifier
# utils:
from ..config.defaults import get_cfg
from ..utils.dataitems import create_data_list_from_file
# models:
from .AttributeClassifier import BaseAttributeClassifier
from .ProgramClassifier import BaseProgramClassifier
from .ProgramClassifierV2 import BaseProgramClassifierV2


def build_classifier(cfg: CfgNode, device: torch.device) -> torch.nn.Module:
    model_name = cfg.MODEL.ARCH
    openset_categories = create_data_list_from_file(cfg.DATA.PATH_TO_CLASSNAMES)
    if model_name == 'BaseAttributeClassifier':
        model = BaseAttributeClassifier(openset_params=cfg.OPENSET, device=device,
                                        openset_categories=openset_categories, model_card_path=cfg.MODEL.BACKBONE)
    elif model_name == 'BaseProgramClassifier':
        model = BaseProgramClassifier(openset_params=cfg.OPENSET, device=device, openset_categories=openset_categories,
                                      model_card_path=cfg.MODEL.BACKBONE)
    elif model_name == 'BaseProgramClassifierV2':
        model = BaseProgramClassifierV2(openset_params=cfg.OPENSET, device=device,
                                        openset_categories=openset_categories, model_card_path=cfg.MODEL.BACKBONE)
    elif model_name == 'LLaVaGptClassifier':
        model = LLaVaGptClassifier(openset_params=cfg.OPENSET, device=device, openset_categories=openset_categories,
                                   config=cfg)
    elif model_name == 'FineTunedAttributeClassifier':
        model = FineTunedAttributeClassifier(openset_params=cfg.OPENSET, device=device, openset_categories=openset_categories,
                                             model_size=cfg.MODEL.MODEL_SIZE, model_card_path=cfg.MODEL.BACKBONE)
    else:
        raise NotImplementedError("Model name is not supported..")
    model.output_type = 'openset'

    print('loading model {} of network type: {} and output_type {}..'.format(model_name, model.__class__.__name__,
                                                                             model.output_type))

    return model



if __name__ == '__main__':
    # from tal.models.losses import load_loss
    # from tal.models.optimizers import load_optimizer
    cfg = get_cfg()

    model_name = cfg.MODEL.ARCH
    model = build_classifier(cfg=cfg)
    model = model.cuda()
    video_input = torch.rand(cfg.TRAIN.BATCH_SIZE, cfg.MODEL.FEAT_DIM, cfg.MODEL.NUM_FRAMES).cuda()
    pred = model(video_input)
    # pred_class = torch.rand(cfg.TRAIN.BATCH_SIZE, cfg.MODEL.NUM_CLASSES).cuda()
    # criterion = load_loss("L2")
    # loss = criterion(pred_kpts, kpts)
    # print(loss.item())
    #
    # optimizer = load_optimizer("Adam", model.parameters(), learningrate=1e-4)
    # optimizer.step()
    #
    print("done")
