import torch
from fvcore.common.config import CfgNode
from typing import List, Tuple, Dict
from tqdm import tqdm

from ..utils.misc import to_numpy, dict_to_numpy
from ..utils.meters import AverageMeter
from ..utils.vis import plot_vqa_prediction_to_screen, plot_vqa_prediction


def train_vanilla(epoch: int,
                  loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  model: torch.nn.Module,
                  device: torch.device,
                  criterion: torch.nn.Module,
                  cfg: CfgNode,
                  prossesID: int = None
                  ) -> Tuple[int, list]:
    """training"""

    model.train()
    prefix = 'Training'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    losses = LossMeters()
    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, data in enumerate(loader, 0):

            # ================= Extract Data ==================
            filenames = data[-1]

            # =================== forward =====================
            batch_loss, batch_output = run_forward(model, data, criterion, cfg, device)

            # =================== backward ====================
            if optimizer is not None:
                optimizer.zero_grad()
                batch_loss["loss"].backward()
                optimizer.step()

            pbar.update()
            for dat_name, dat in batch_output.items():
                to_numpy(dat)

            # accumulate losses:
            losses.accumulate(batch_loss=batch_loss["reported"], num_files=len(filenames))

    return losses.get_losses()


@torch.no_grad()
def test(mode: str,
         epoch: int,
         loader: torch.utils.data.DataLoader,
         model: torch.nn.Module,
         device: torch.device,
         criterion: torch.nn.Module,
         cfg: CfgNode,
         prossesID: str = None,
         results_saver = None) -> Tuple[Dict, Dict, Dict]:
    """perform validation  / test epoch

    Returns
    -------
    `tuple` (outputs,inputs)
    """
    model.eval()
    if mode == 'validation':
        prefix = 'Validating'
    elif mode == 'test':
        prefix = 'Testing'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    losses = LossMeters()

    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, data in enumerate(loader, 0):
            # =================== forward =====================
            batch_loss, batch_output = run_forward(model, data, criterion, cfg, device)

            pbar.update()
            # accumulate outputs (intervals times and scores) and inputs (GT times and scores):
            filenames = data[2]
            gt_labels = data[3]
            results_saver.append((filenames,gt_labels,batch_output))




########################################
########################################
# Loss aux:
########################################
########################################
class LossMeters():
    def __init__(self, loss_names=None):

        self.batch_list = []
        self.losses = None
        if loss_names is not None:
            self.losses = self._build_meters(loss_names)

    def _build_meters(self, loss_names):
        self.losses = dict()
        for loss_name in loss_names:
            self.losses[loss_name] = AverageMeter()

    def accumulate(self, batch_loss, num_files):
        if self.losses is None:
            loss_names = list(batch_loss.keys())
            self._build_meters(loss_names)
        for loss_name, loss_values in batch_loss.items():
            self.losses[loss_name].update(loss_values, num_files)

    def get_losses(self):
        return self.losses


########################################
########################################
# Forward methods:
########################################
########################################

def run_forward(model: torch.nn.Module, data: List, criterion: torch.nn.Module, cfg: CfgNode, device: torch.device) -> \
Tuple[Dict, Dict]:
    """
    :param model:
    :param data: minibatch
    :param cfg:
    :param device:
    :return:
    """

    # ================= Extract Data ==================

    labels = data[1].to(device)  # .long()    # prepare category in int-64bit
    images = data[0].to(device)  # .float()   # prepare features in float-32bit
    images_paths = data[2]
    # preprocessed = torch.stack([model.preprocess(image) for image in images])
    # images = preprocessed.to(device)

    # =================== forward =====================
    image_labels_similarity, explanations = model(images,images_paths)  # , labels)
    # descr_predictions = image_labels_similarity.argmax(dim=1)

    batch_output = {"image_labels_similarity": image_labels_similarity, "explanations": explanations}

    # =================== Loss ====================
    batch_loss = 0  # criterion(batch_output, image_category)

    return batch_loss, batch_output

