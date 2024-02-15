import shutil
import os
import datetime
import pandas as pd
import numpy as np
from ovod.utils.misc import to_numpy, dict_to_numpy
import ovod.utils.logger as logging
from evaluation.metrics import calc_topk_accuracy, binary_classification_correctness, build_confision_matrix

logger = logging.get_logger(__name__)


def build_results_saver(cfg, model, test_info):
    model_name = cfg.MODEL.ARCH
    if 'RESULTS_SAVER' in cfg and cfg.RESULTS_SAVER == 'SeverResultSaver':
        results_saver = SeverResultSaver(cfg, model, cfg.SAVE_RESULTS_EVERY, test_info)
    elif model_name in ['BaseAttributeClassifier', 'FineTunedAttributeClassifier']:
        results_saver = DefaultResultsSaver(cfg, model, cfg.SAVE_RESULTS_EVERY, test_info)
    elif model_name == 'LLaVaGptClassifier':
        results_saver = LLaVaResultsSaver(cfg, model, cfg.SAVE_RESULTS_EVERY, test_info)
    else:
        raise NotImplementedError("Model name is not supported..")
    return results_saver


class DefaultResultsSaver:
    def __init__(self, cfg, model, save_every=5, test_info=''):
        self.cfg = cfg
        self.save_every = save_every
        self.model = model
        self.results = []
        self.plots_amount = cfg.TEST.NUM_DEMO_PLOTS
        self.set_output_paths(cfg, test_info)
        os.makedirs(self.out_directory, exist_ok=True)
        self.csv_output_path = os.path.join(self.out_directory, 'results.csv')

        self.test_info = test_info
        # write top1 and top5 to stats file:
        self.stats_file = os.path.join(self.out_directory, 'stats.txt')

    def set_output_paths(self, cfg, test_info):
        dataset_name, subset, arch = cfg.TEST.DATASET, cfg.TEST.SUBSET, cfg.MODEL.ARCH
        openset = cfg.DATA.PATH_TO_CLASSNAMES.split("/")[-2].replace("/", "")
        if cfg.MODEL.BACKBONE == "" or cfg.MODEL.BACKBONE is None:
            backbone = ""
        else:
            backbone = (os.path.splitext(os.path.basename(cfg.MODEL.BACKBONE))[0]).replace('-', '_')
        description_type = os.path.splitext(os.path.basename(cfg.OPENSET.ATTRIBUTES_FILE))[0]
        description_type = description_type.split("descriptions_")[1]
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_directory = os.path.join(cfg.OUTPUT_DIR, dataset_name, openset, arch, backbone, description_type,
                                          time_stamp)

    def append(self, result):
        self.results.append(result)
        if len(self.results) % self.save_every == 0:
            self.save()

    def parse_results_to_elegant_form(self):
        outputs = {}
        for filenames, gt_labels, batch_output in self.results:
            for ii, filename in enumerate(filenames):
                self.parse_one_line(batch_output, filename, gt_labels, ii, outputs)
        return outputs

    def parse_one_line(self, batch_output, filename, gt_labels, ii, outputs):
        desc_predictions = batch_output["image_labels_similarity"][ii]
        explantions = batch_output["explanations"][ii]
        label_prediction = desc_predictions.argmax().cpu().numpy()
        # label_gt = labels[ii].cpu().numpy()
        outputs[filename] = {"desc_predictions": {self.model.index2class[ii]: sc for ii, sc in
                                                  enumerate(desc_predictions.detach().cpu().numpy())},
                             "label_pred": self.model.index2class[int(label_prediction)],
                             "label_gt": gt_labels[ii],
                             "explanation_pred": dict_to_numpy(explantions[label_prediction]) if type(
                                 explantions[label_prediction]) == dict else explantions[
                                 label_prediction].detach().cpu().numpy(),
                             "explanation_gt": dict_to_numpy(
                                 explantions[self.model.class2index[gt_labels[ii]]]) if type(
                                 explantions[self.model.class2index[gt_labels[ii]]]) is dict else explantions[
                                 self.model.class2index[gt_labels[ii]]].detach().cpu().numpy()
                             }

    def save(self):
        df = self.create_df()
        df.to_csv(self.csv_output_path, index=False)  # , header=(not os.path.exists(self.csv_output_path)))
        top1, top5 = self.get_top_for_df(df)

        logger.info(f"Current top-1: {top1}")
        logger.info(f"OV-OC model predictions were stored in {self.csv_output_path}")

    def create_df(self):
        # Call to father with create_df
        test_outputs = self.parse_results_to_elegant_form()
        results = [res["desc_predictions"] for res in test_outputs.values()]
        true_labels = [res["label_gt"] for res in test_outputs.values()]
        df = pd.DataFrame(results)
        # df_numbers = df.loc[:, df.columns != 'filename']
        max = df.max(axis=1)
        argmax = df.idxmax(axis=1)
        df.insert(0, "score", max)
        df.insert(0, "pred", argmax)
        df.insert(0, "gt", true_labels)
        filenames = [fname.replace(self.cfg.DATA.ROOT_DIR, "") for fname in test_outputs.keys()]
        df.insert(0, "filename", filenames)
        return df

    def calc_metrics(self):
        df = self.create_df()
        # iterate over rows, and return the top 5 predictions for each row:
        top1, top5 = self.get_top_for_df(df)

        # add statistics about binary classification and confusion matrix
        binary_results, multi_class_correctness, binary_out_prints = binary_classification_correctness(df,
                                                                                                       categories=None)
        results, confusion_matrix, confusion_out_prints = build_confision_matrix(df, self.out_directory)

        with open(self.stats_file, 'a') as f:
            f.write(f"{self.test_info}\n")
            f.write(f"Top-1 accuracy: {top1}\n")
            f.write(f"Top-5 accuracy: {top5}\n")
            for p in binary_out_prints[:-1]:
                f.write(f"{p}\n")
            for p in confusion_out_prints[1:]:
                f.write(f"{p}\n")
        logger.info(f"Top-1 and Top-5 accuracies, and other statistical measures were written to {self.stats_file}")

        return top1, top5

    def get_top_for_df(self, df):
        max_top_k = 10
        gt = df['gt']
        preds = self.get_only_numerical_predictions_of_df(df)

        preds_sorted = preds.apply(lambda x: x.nlargest(max_top_k).index.tolist(), axis=1)
        #
        top1 = calc_topk_accuracy(preds_sorted, gt, k=1)
        logger.info(f"Top-1 accuracy: {top1}")
        top5 = calc_topk_accuracy(preds_sorted, gt, k=5)
        logger.info(f"Top-5 accuracy: {top5}")
        return top1, top5

    def get_top(self):
        df = self.create_df()
        top1, top5 = self.get_top_for_df(df)
        return top1, top5

    def get_only_numerical_predictions_of_df(self, df):
        return df.iloc[:, 4:]

    def visualize(self):
        test_outputs = self.parse_results_to_elegant_form()
        if self.plots_amount > 0:
            logger.info("Plotting results...")
            plot_folder = os.path.join(self.out_directory, 'plots')
            # delete plot_folder if exists:
            if os.path.exists(plot_folder):
                shutil.rmtree(plot_folder)
            os.makedirs(plot_folder, exist_ok=True)
            # select random keys from test_outputs:
            keys = list(test_outputs.keys())
            random_keys = np.random.choice(keys, self.plots_amount, replace=False)
            for img in random_keys:
                label_pred = test_outputs[img]["label_pred"]
                label_gt = test_outputs[img]["label_gt"]
                self.add_model_atts_for_visualization(img, label_gt, label_pred, test_outputs)
                self.model.plot_prediction(img, test_outputs[img], plot_folder)
                # plot_attributes_predictions(img, test_outputs[img], plot_folder)  # df, cfg, out_directory)
            logger.info(f"Done plotting results. {self.plots_amount} plots were saved in {plot_folder}")

    def add_model_atts_for_visualization(self, img, label_gt, label_pred, test_outputs):
        test_outputs[img]["attrs_pred"] = self.model.attributes[label_pred]["queries"]
        test_outputs[img]["attrs_gt"] = self.model.attributes[label_gt]["queries"]


class LLaVaResultsSaver(DefaultResultsSaver):
    def parse_one_line(self, batch_output, filename, gt_labels, ii, outputs):
        desc_predictions = batch_output["image_labels_similarity"][ii]
        explanations = batch_output["explanations"][ii]
        label_prediction = desc_predictions.argmax().cpu().numpy()
        outputs[filename] = {"desc_predictions": {self.model.index2class[ii]: sc for ii, sc in
                                                  enumerate(desc_predictions.detach().cpu().numpy())},
                             "label_pred": self.model.index2class[int(label_prediction)],
                             "label_gt": gt_labels[ii],
                             "question": explanations['question'],
                             "answer": explanations['answer']}

    def create_df(self):
        df = super().create_df()
        test_outputs = self.parse_results_to_elegant_form()
        questions = [res["question"] for res in test_outputs.values()]
        answers = [res["answer"] for res in test_outputs.values()]
        df.insert(len(df.columns), "question", questions)
        df.insert(len(df.columns), "answer", answers)
        return df

    def add_model_atts_for_visualization(self, img, label_gt, label_pred, test_outputs):
        pass

    def get_only_numerical_predictions_of_df(self, df):
        return df.iloc[:, 4:-2]

    def set_output_paths(self, cfg, test_info):
        dataset_name, subset, arch = cfg.TEST.DATASET, cfg.TEST.SUBSET, cfg.MODEL.ARCH
        openset = cfg.DATA.PATH_TO_CLASSNAMES.split("/")[-2].replace("/", "")
        # Take only the name of the file
        prompt_question_type = os.path.splitext(os.path.basename(cfg.MODEL.QUESTIONS_PATH))[0]
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_directory = os.path.join(cfg.OUTPUT_DIR, dataset_name, openset, arch, prompt_question_type, time_stamp)


class SeverResultSaver(DefaultResultsSaver):
    def save(self):
        pass

    def visualize(self):
        pass

    def calc_metrics(self):
        pass