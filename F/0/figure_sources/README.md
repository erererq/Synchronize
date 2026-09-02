# Figure source map

Run `python generate_all_figures.py` to rebuild reproducible counterparts of all manuscript figures in `reproduced_figures/`.
The `FIGURE_MAP` dictionary provides a one-to-one mapping from Fig. 1 through
Fig. 18 to a dedicated Python function. Fixed random seeds are used throughout.
The original figure files are intentionally never overwritten. The reviewer-requested
two-dimensional heat maps and transmitter--receiver flowchart are included as
additional manuscript figures alongside the retained originals.
