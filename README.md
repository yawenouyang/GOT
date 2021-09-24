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

Unknown intent detection:
```bash
bash train.sh
```

Generate OOD utteranes:
```bash
bash locating.sh
bash generating.sh
bash weighting.sh
```

## Citation
```bibtex
@inproceedings{ouyang-etal-2021-energy,
    title = "Energy-based Unknown Intent Detection with Data Manipulation",
    author = "Ouyang, Yawen  and
      Ye, Jiasheng  and
      Chen, Yu  and
      Dai, Xinyu  and
      Huang, Shujian  and
      Chen, Jiajun",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.252",
    doi = "10.18653/v1/2021.findings-acl.252",
    pages = "2852--2861",
}
```