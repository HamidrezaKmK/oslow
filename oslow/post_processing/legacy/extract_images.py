"""
This is a script taken from https://gist.github.com/hysts/81a0d30ac4f33dfa0c8859383aec42c2
It is used to extract images from a Tensorboard event file.
"""

import argparse
import pathlib
import numpy as np
import cv2
from tensorboard.backend.event_processing import event_accumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    event_acc = event_accumulator.EventAccumulator(args.path, size_guidance={"images": 0})
    event_acc.Reload()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    for tag in event_acc.Tags()["images"]:
        events = event_acc.Images(tag)

        tag_name = tag.replace("/", "_")
        dirpath = outdir / tag_name
        dirpath.mkdir(exist_ok=True, parents=True)

        for index, event in enumerate(events):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(s, cv2.IMREAD_COLOR)
            outpath = dirpath / "{:04}.jpg".format(index)
            cv2.imwrite(outpath.as_posix(), image)


if __name__ == "__main__":
    main()
