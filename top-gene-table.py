#!/usr/bin/env python

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# python script_name.py -f your_file.tsv -un -nt 10 -o output.png
# 用于已经有了基因和置信度信息，然后用来生成对应table的

# Argument parser
prs = argparse.ArgumentParser(description="Process some data.")
prs.add_argument("-f", "--file_name",  help="full path to file", required=True)
prs.add_argument("-un", "--use_numbers", help="add left column with number of item", action='store_true')
prs.add_argument("-nt", "--n_top", help="number of top genes to select", type=float)
prs.add_argument("-o", "--out_name", help="full path to output", default=None)

args = prs.parse_args()

if args.out_name is None:
    args.out_name = "top-genes.png"

# Read data ? 
data = pd.read_csv(args.file_name, sep='\t', index_col=0)

# Add rank column
if args.use_numbers:
    data['Rank'] = range(1, len(data) + 1)

data = data[['Rank', 'Gene', 'Coefficient']]
data['Coefficient'] = round(data['Coefficient'], 5)

# Select top genes
if args.n_top is not None:
    data = data.head(int(args.n_top))

# Save the table as a PNG image
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
tbl = table(ax, data, loc='center', colWidths=[0.1] * len(data.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)

plt.savefig(args.out_name, bbox_inches='tight')
plt.close()
