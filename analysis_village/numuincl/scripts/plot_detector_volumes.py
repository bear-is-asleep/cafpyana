#!/usr/bin/env python3
"""Generate YZ/XZ diagrams of SBND AV, fiducial, and rejection volumes."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
CAFPYANA_WD = REPO_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(CAFPYANA_WD)):
    if p not in sys.path:
        sys.path.insert(0, p)

SBNDANA_DIR = REPO_ROOT / 'sbnd'
sys.path.insert(0, str(SBNDANA_DIR))
sys.path.insert(0, str(REPO_ROOT))
plt.style.use(str(SBNDANA_DIR / 'plotlibrary' / 'numu2025.mplstyle'))

from sbnd.detector.volume_plots import plot_xz_overview, plot_yz_tpc

DEFAULT_OUT_DIR = SBNDANA_DIR / 'plots' / 'detector_volumes'

FIGURES = (
    ('detector_xz_overview.png', 'xz', None),
    ('detector_yz_tpc0_west.png', 'yz', 0),
    ('detector_yz_tpc1_east.png', 'yz', 1),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f'output directory (default: {DEFAULT_OUT_DIR})',
    )
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--show', action='store_true', help='show figures interactively')
    parser.add_argument('--no-annotations', action='store_true')
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    show_annotations = not args.no_annotations

    saved: list[Path] = []
    for filename, view, tpc_index in FIGURES:
        plt.close('all')
        if view == 'xz':
            fig = plot_xz_overview(show_annotations=show_annotations)
        else:
            fig = plot_yz_tpc(tpc_index, show_annotations=show_annotations)
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        saved.append(out_path)
        print(out_path)

    if args.show:
        plt.show()
    else:
        plt.close('all')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
