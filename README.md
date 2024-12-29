# e4e training code for FACEMUG and [Visual Style Prompt Restoration](https://github.com/LonglongaaaGo/VSPBFR)


- Note: this project is from the paper
``Designing an Encoder for StyleGAN Image Manipulation (SIGGRAPH 2021)''
- We just use it to train style encoders. For more technique problems, please refer to e4e.



## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### Installation
- Clone the repository:
``` 
git clone https://github.com/omertov/encoder4editing.git
cd encoder4editing
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/e4e_env.yaml`.

### Inference Notebook
We provide a Jupyter notebook found in `notebooks/inference_playground.ipynb` that allows one to encode and perform several editings on real images using StyleGAN.   

### Pretrained Models
Please download the pre-trained models from the following links. Each e4e model contains the entire pSp framework architecture, including the encoder and decoder weights.
| Path | Description
| :--- | :----------
|[FFHQ Inversion](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing)  | FFHQ e4e encoder.
|[Cars Inversion](https://drive.google.com/file/d/17faPqBce2m1AQeLCLHUVXaDfxMRU2QcV/view?usp=sharing)  | Cars e4e encoder.
|[Horse Inversion](https://drive.google.com/file/d/1TkLLnuX86B_BMo2ocYD0kX9kWh53rUVX/view?usp=sharing)  | Horse e4e encoder.
|[Church Inversion](https://drive.google.com/file/d/1-L0ZdnQLwtdy6-A_Ccgq5uNJGTqE7qBa/view?usp=sharing) | Church e4e encoder.

If you wish to use one of the pretrained models for training or inference, you may do so using the flag `--checkpoint_path`.

In addition, we provide various auxiliary models needed for training your own e4e model from scratch.
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during training.
|[MOCOv2 Model](https://drive.google.com/file/d/18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe/view?usp=sharing) | Pretrained ResNet-50 model trained using MOCOv2 for use in our simmilarity loss for domains other then human faces during training.

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`. However, you may use your own paths by changing the necessary values in `configs/path_configs.py`. 

## Training
To train the e4e encoder, make sure the paths to the required models, as well as training and testing data is configured in `configs/path_configs.py` and `configs/data_configs.py`.



#### **Training the e4e Encoder for the FACEMUG**
For the data preparation, please refer to [FACEMUG](https://github.com/LonglongaaaGo/FACEMUG_trainer).
```
python FACEMUG_encoder_trainer.py
--dataset_type ffhq_encode
--exp_dir new/experiment/directory
--start_from_latent_avg
--use_w_pool
--w_discriminator_lambda 0.1
--progressive_start 20000
--id_lambda 0.5
--val_interval 10000
--max_steps 200000
--stylegan_size 1024
--workers 8
--batch_size 8
--test_batch_size 4
--test_workers 4
--in_channel 26
--save_training_data
--keep_optimizer
--multi_modal True
--checkpoint_path [pre-trained FFHQ e4e encoder, please find it in the above pre-trained models]
--img_path [image folder path]
--edge_root [sketch folder path]
--semantic_root [semantic folder path]
--color_root [color folder path]
--img_test_path [image folder path for testing]
--edge_test_root [sketch folder path for testing]
--semantic_test_root [semantic folder path for testing]
--color_test_root [color folder path for testing]
```


#### **Training the e4e Encoder for the [Visual Style Prompt Restoration](https://github.com/LonglongaaaGo/VSPBFR)**
```
python Visual_prompt_trainer.py
--dataset_type ffhq_encode
--exp_dir new/experiment/directory
--start_from_latent_avg
--use_w_pool
--w_discriminator_lambda 0.1
--progressive_start 20000
--id_lambda 0.5
--val_interval 10000
--max_steps 200000
--stylegan_size 1024
--workers 8
--batch_size 8
--test_batch_size 4
--test_workers 4
--in_channel 3
--save_training_data
--keep_optimizer
--checkpoint_path [pre-trained FFHQ e4e encoder, please find it in the above pre-trained models]
--img_path [image folder path]
--img_test_path [image folder path for testing]
```




## Acknowledgments
We thank the [encoder4editing](https://github.com/omertov/encoder4editing) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) 

## Citation
If you use this code for your research, please cite their paper <a href="https://arxiv.org/abs/2102.02766">Designing an Encoder for StyleGAN Image Manipulation</a>:


```
@article{tov2021designing,
  title={Designing an Encoder for StyleGAN Image Manipulation},
  author={Tov, Omer and Alaluf, Yuval and Nitzan, Yotam and Patashnik, Or and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2102.02766},
  year={2021}
}

```

Our related papers:

```
@ARTICLE{FACEMUG,
  author={Lu, Wanglong and Wang, Jikai and Jin, Xiaogang and Jiang, Xianta and Zhao, Hanli},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={FACEMUG: A Multimodal Generative and Fusion Framework for Local Facial Editing}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Facial features;Semantics;Codes;Generators;Image synthesis;Faces;Image color analysis;Generative adversarial networks;image-toimage translation;multimodal fusion;image editing;facial editing},
  doi={10.1109/TVCG.2024.3434386}}


@article{LU2025128996,
title = {Do inpainting yourself: Generative facial inpainting guided by exemplars},
journal = {Neurocomputing},
volume = {617},
pages = {128996},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.128996},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224017673},
author = {Wanglong Lu and Hanli Zhao and Xianta Jiang and Xiaogang Jin and Yong-Liang Yang and Kaijie Shi},
keywords = {Generative adversarial networks, Image generation, Image inpainting, Facial image inpainting},
abstract = {We present EXE-GAN, a novel exemplar-guided facial inpainting framework using generative adversarial networks. Our approach not only preserves the quality of the input facial image but also completes the image with exemplar-like facial attributes. We achieve this by simultaneously leveraging the global style of the input image, the stochastic style generated from the random latent code, and the exemplar style of the exemplar image. We introduce a novel attribute similarity metric to encourage the networks to learn the style of facial attributes from the exemplar in a self-supervised way. To guarantee the natural transition across the boundaries of inpainted regions, we introduce a novel spatial variant gradient backpropagation technique to adjust the loss gradients based on the spatial location. We extensively evaluate EXE-GAN on public CelebA-HQ and FFHQ datasets with practical applications, which demonstrates the superior visual quality of facial inpainting. The source code is available at https://github.com/LonglongaaaGo/EXE-GAN.}
}

@article{LU2024111312,
title = {Visual style prompt learning using diffusion models for blind face restoration},
journal = {Pattern Recognition},
pages = {111312},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.111312},
url = {https://www.sciencedirect.com/science/article/pii/S003132032401063X},
author = {Wanglong Lu and Jikai Wang and Tao Wang and Kaihao Zhang and Xianta Jiang and Hanli Zhao},
keywords = {Denoising diffusion probabilistic models, Generative adversarial networks, Blind face restoration},
abstract = {Blind face restoration aims to recover high-quality facial images from various unidentified sources of degradation, posing significant challenges due to the minimal information retrievable from the degraded images. Prior knowledge-based methods, leveraging geometric priors and facial features, have led to advancements in face restoration but often fall short of capturing fine details. To address this, we introduce a visual style prompt learning framework that utilizes diffusion probabilistic models to explicitly generate visual prompts within the latent space of pre-trained generative models. These prompts are designed to guide the restoration process. To fully utilize the visual prompts and enhance the extraction of informative and rich patterns, we introduce a style-modulated aggregation transformation layer. Extensive experiments and applications demonstrate the superiority of our method in achieving high-quality blind face restoration.}
}

```

