# ZeroSlide: Is Zero-Shot Classification Adequate for Lifelong Learning in Whole-Slide Image Analysis in the Era of Pathology Vision-Language Foundation Models?

This repository provides the official implementation of the paper:

> "ZeroSlide: Is Zero-Shot Classification Adequate for Lifelong Learning in Whole-Slide Image Analysis in the Era of Pathology Vision–Language Foundation Models?"

![{8EA5F97D-BFA0-4E7D-858B-E7B0D4277301}](https://hackmd.io/_uploads/HJuSd1H1bl.png)

# 1. Get started

ZeroSlide leverages the TITAN foundation model to reformulate zero-shot learning as a task-incremental continual learning problem, enabling direct comparison against continual learning–based methods for Whole Slide Image (WSI) classification.
> Note: Access to the TITAN model on Hugging Face must be granted by its authors.

Before running the code, install the required dependency:

```
pip install transformers
```

This implementation has been tested under the following environment:

| Library     | Version      |
| ----------- | ------------ |
| Python      | 3.11.11      |
| PyTorch     | 2.3.0+cu121  |
| Torchvision | 0.18.0+cu121 |

# 2. Running Inference and Obtaining Results

All test slide embeddings are stored as *.pt files under: `./slide_feats_for_zeroshot/`

After installing `torch` and `transformers`, and ensuring access to the TITAN model, run:

```
python titan_zeroshot.py
```

# 3. Citation

If you find this work useful, please consider citing:

```
@article{bui2025zeroslide,
  title={ZeroSlide: Is Zero-Shot Classification Adequate for Lifelong Learning in Whole-Slide Image Analysis in the Era of Pathology Vision-Language Foundation Models?},
  author={Bui, Doanh C and Pham, Hoai Luan and Le, Vu Trung Duong and Vu, Tuan Hai and Tran, Van Duy and Nakashima, Yasuhiko},
  journal={arXiv preprint arXiv:2504.15627},
  year={2025}
}
```
