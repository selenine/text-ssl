# text-ssl

This is an experimental repo that I created to probe what would happen if I tried to use self-supervised representation learning techniques on textual data, a la DINOv3

## Quick Start

```bash
uv sync
uv run accelerate launch --config_file accelerate.yaml -m text_ssl.cli
```
