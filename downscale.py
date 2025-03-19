import cv2
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scale',  type=int, default=2,         help='Downscale factor'  )
parser.add_argument('-i', '--input',  type=str, default='inputs',  help='Input folder'      )
parser.add_argument('-o', '--output', type=str, default='results', help='Output folder'     )
args = parser.parse_args()
try:
    print(f"Writing downscaled images in folder: {args.output}...")
    Path(args.output).mkdir(parents=True, exist_ok=False)
except Exception as e:
    print(f"{e} - Skipping.")
for imgpath in Path(args.input).rglob('*.png'):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (0,0), fx=1/args.scale, fy=1/args.scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"./{args.output}/{imgpath.stem}{imgpath.suffix}", img=img)