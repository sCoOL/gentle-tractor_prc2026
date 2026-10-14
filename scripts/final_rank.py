"""The rank-population size the layer builds assert against.

The earlier Jan/Jul export had 344,841 departures and the layer scripts hard-coded that number; the final file has 670,790
(Jan + Feb + Jun + Jul). `n_rank()` is the length of the file the chain runs on: by default the template
data/submitting.parquet; tools/final_chain.py sets `N` to the length of its v45 input (670,790).
"""
import pandas as pd

import common

N = None


def n_rank() -> int:
    global N
    if N is None:
        N = len(pd.read_parquet(common.DATA / 'submitting.parquet', columns=['MVT_ID_mvt']))
    return N
