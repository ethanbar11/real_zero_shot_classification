import os.path
import os
import pandas as pd
import math
import json
import numpy as np
from fvcore.common.config import CfgNode

def csv_creation(test_outputs: dict):
    # create csv with categories and scores
    results = [res["desc_predictions"] for res in test_outputs.values()]
    true_labels = [res["label_gt"] for res in test_outputs.values()]
    df = pd.DataFrame(results)
    #df_numbers = df.loc[:, df.columns != 'filename']
    max = df.max(axis=1)
    argmax = df.idxmax(axis=1)
    df.insert(0, "score", max)
    df.insert(0, "pred", argmax)
    df.insert(0, "gt", true_labels)
    return df
