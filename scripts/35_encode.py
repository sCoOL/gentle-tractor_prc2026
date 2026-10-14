import argparse, importlib.util, pathlib
spec = importlib.util.spec_from_file_location('encodings_', pathlib.Path(__file__).with_name('encodings.py')); enc = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)
p = argparse.ArgumentParser(); p.add_argument('--force', action='store_true'); p.add_argument('--no-unseen-map', action='store_true'); a = p.parse_args(); enc.build(force=a.force, use_unseen_map=not a.no_unseen_map)
