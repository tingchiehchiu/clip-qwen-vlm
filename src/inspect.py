"""Load the two open component checkpoints and inspect bridge dimensions/scopes."""

import argparse
import json

from .common import read_json
from .model import ConstructedLVLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/construction.json")
    args = parser.parse_args()
    model = ConstructedLVLM(read_json(args.config))
    report = {
        "adapter_type": model.config_values["adapter_type"],
        "vision_hidden_size": model.vision.config.hidden_size,
        "language_hidden_size": model.language.config.hidden_size,
        "adapter_parameters": sum(p.numel() for p in model.adapter.parameters()),
        "visual_tokens": model.visual_tokens,
        "random_trainable": model.freeze("random"),
        "alignment_trainable": model.freeze("align"),
        "instruction_trainable": model.freeze("instruct"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
