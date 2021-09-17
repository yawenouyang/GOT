# GOT

This repository (under construction) is the implementation of 《Energy-based Unknown Intent Detection with Data Manipulation》, which is accepted by Findings of ACL, 2021.


## Requirements
Install requirements:
```bash
pip install -r requirements.txt
```

Other preparations:
- Download [bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main), [gpt2-base](https://huggingface.co/gpt2/tree/main) and put them to the `pretrained` folder. You can also customize their location by modifying the `*.yaml` files in the `configs` folder.  


## Usage
Generate OOD utteranes:
```bash
bash locating.sh
bash generating.sh
bash weighting.sh
```

Run energy score:
```bash
bash train.sh
```

Enjoy :)