import argparse, ext_weather
p = argparse.ArgumentParser(); p.add_argument("--force", action="store_true"); a = p.parse_args(); ext_weather.build(force=a.force)
