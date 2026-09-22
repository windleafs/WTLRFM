"""Merge the non-homoc decoupled cache records with the v4 calibration cache.

Symlinks existing npz files and writes one merged manifest so training sees
480 counterfactual records (360 v2 variants + 120 v4 calibration variants).
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dec-cache', type=Path,
                   default=Path('/data/zhuangyang/geometry_flow_v2_dec_cache'))
    p.add_argument('--calib-cache', type=Path,
                   default=Path('/data/zhuangyang/geometry_flow_v2_calib_cache'))
    p.add_argument('--out', type=Path,
                   default=Path('/data/zhuangyang/geometry_flow_v2_decv4_cache'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dec = json.loads((args.dec_cache/'manifest.json').read_text())
    calib = json.loads((args.calib_cache/'manifest.json').read_text())
    keep = [r for r in dec['records'] if not r['id'].endswith('_homoc')]
    records = []
    for src_dir, subset in ((args.dec_cache, keep),
                            (args.calib_cache, calib['records'])):
        for r in subset:
            target = args.out/r['cache_file']
            if not target.exists():
                target.symlink_to(src_dir/r['cache_file'])
            records.append(r)
    merged = dict(dec)
    merged['records'] = records
    merged['note'] = ('v4 mixture: 360 decoupled v2 variants (reseed/texture/'
                      'cswap) + 120 v4 calibration variants (calibhom/calibuni)')
    (args.out/'manifest.json').write_text(json.dumps(merged, indent=2)+'\n')
    counts = {}
    for r in records:
        tag = r['id'].rsplit('_', 1)[-1]
        counts[tag] = counts.get(tag, 0) + 1
    print(f'merged {len(records)} records -> {args.out}; counts={counts}')


if __name__ == '__main__':
    main()
