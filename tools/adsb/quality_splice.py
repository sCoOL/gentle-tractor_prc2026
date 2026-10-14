"""Keep the reference 2025 rows of adsb_quality.parquet (run after every `121 quality`, before `121 features` /
`255 features`).

`121 quality` breaks callsign ties with max(set(names), key=names.count): on a tie the winner depends on Python's
per-process string-hash seed, so a rerun moves a few 2025 arrivals per airport-month. The 2025 airport-months depend only
on 2025 inputs, and the reference values are the ones every saved 2025 frame / model was built with. So the table keeps
the reference 2025 rows verbatim and the freshly computed 2026 rows. The computed table is kept as
adsb_quality_computed.parquet. Without the reference file (PRC_FROZEN/ref_adsb/adsb_quality.parquet) the computed table
is left as it is (2025 rows then differ from the reference only by those tie-breaks).

  py -3.14 tools/adsb/quality_splice.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import prc_config  # noqa: E402

R = prc_config.ADSB_DIR
S = prc_config.FROZEN / 'ref_adsb' / 'adsb_quality.parquet'

if not S.exists():
    print(f'no reference {S}: adsb_quality.parquet left as computed')
    sys.exit(0)
comp = pd.read_parquet(R / 'adsb_quality.parquet')
comp.to_parquet(R / 'adsb_quality_computed.parquet', index=False)
ref = pd.read_parquet(S)
m25 = ref[ref.month.astype(str) < '2026']
c25 = comp[comp.month.astype(str) < '2026']
ka = set(zip(c25.APT.astype(str), c25.month.astype(str)))
kb = set(zip(m25.APT.astype(str), m25.month.astype(str)))
assert ka == kb, f'2025 airport-months differ: only computed {sorted(ka - kb)}, only reference {sorted(kb - ka)}'
assert list(ref.columns) == list(comp.columns), (list(ref.columns), list(comp.columns))
out = pd.concat([m25.astype(comp.dtypes.to_dict()), comp[comp.month.astype(str) >= '2026']], ignore_index=True)
out = out.sort_values(['APT', 'month'], kind='stable').reset_index(drop=True)
out.to_parquet(R / 'adsb_quality.parquet', index=False)
print(f'adsb_quality.parquet: {len(m25)} reference 2025 rows + {int((comp.month.astype(str) >= "2026").sum())} computed 2026 rows')
