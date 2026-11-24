"""Use the production room65 surface builder in a private directory for a selected RGB PLY."""
import argparse
from pathlib import Path
import runpy
import sys

here=Path(__file__).resolve().parent
base=here.parents[1]/'scenes/room65_selected_anchored'
parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('rgb',type=Path);args=parser.parse_args()
out=args.output.resolve();out.mkdir(exist_ok=False)
for name in ('models','inputs','collision','checks'):(out/name).mkdir()
(out/'models/rgb.ply').symlink_to(args.rgb.resolve(strict=True));(out/'models/thermal.ply').symlink_to(base/'models/thermal.ply')
for src in (base/'inputs').iterdir():
    if src.name!='frame.json':(out/'inputs'/src.name).symlink_to(src)
sys.path.insert(0,str(base));import room
room.ROOT=out
runpy.run_path(str(base/'prepare.py'),run_name='__main__')
