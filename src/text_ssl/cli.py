import argparse
from pathlib import Path

from text_ssl.configs import ROOT, load_configs
from text_ssl.nn.model import Transformer
from text_ssl.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    model_cfg, train_cfg = load_configs(args.config)
    train(Transformer(model_cfg), train_cfg)


if __name__ == "__main__":
    main()
