import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zeroshot.utils.parser import parse_args, load_config
from zeroshot.config.defaults import assert_and_infer_cfg
from zeroshot.datasets.build import build_dataset

if __name__ == '__main__':
    """
    Main function for plotting few data items for datasets. 
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    ds_name = cfg.TRAIN.DATASET
    g = build_dataset(ds_name, cfg, "trainval")
    g = g.datasets[0]

    basename = os.path.splitext(os.path.basename(args.cfg_file))[0]
    plot_directory = os.path.join(cfg.OUTPUT_DIR, "visu/", basename)    #ds_name)
    os.makedirs(plot_directory, exist_ok=True)

    for k in range(cfg.SHOWDATA.RANGE_START, min(len(g), cfg.SHOWDATA.RANGE_END), cfg.SHOWDATA.STEP):
        if cfg.SHOWDATA.PLOT:
            g.plot_item(index=k, plot_directory=plot_directory)
        if cfg.SHOWDATA.CHECK:
            try:
                data = g.get_img_and_meta(index=k)
                print("vid id {} from {} [{}] is valid".format(data["vid_id"], data["vid_name"], data["vid_link"]))
            except AssertionError as msg:
                #print("{} is not valid, assertion failed".format(data["filename"]))
                print(msg)
