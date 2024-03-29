# CateNorm

## Paper
This repository provides the official implementation of training CateNorm in the following paper:

<b>CateNorm: Categorical Normalization for Robust Medical Image Segmentation</b> <br/>
[Junfei Xiao](https://lambert-x.github.io/)<sup>1</sup>, [Lequan Yu](https://yulequan.github.io/)<sup>2</sup>, [Zongwei Zhou](https://www.zongweiz.com/)<sup>1</sup>, [Yutong Bai](https://scholar.google.com/citations?user=N1-l4GsAAAAJ&hl=en)<sup>1</sup>,  <br/>
[Lei Xing](https://profiles.stanford.edu/lei-xing)<sup>3</sup>, [Alan Yuille](https://scholar.google.com/citations?user=FJ-huxgAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Yuyin Zhou](https://yuyinzhou.github.io/)<sup>4</sup> <br/>
<sup>1 </sup>Johns Hopkins University,   <sup>2 </sup>The University of Hong Kong, <br/>
<sup>3 </sup>Stanford University,   <sup>4 </sup>UC Santa Cruz <br/>
MICCAI Workshop on Domain Adaptation and Representation Transfer (DART), 2022 <br/>
${\color{red} {\textbf{Best Paper Award Honourable Mention}}}$ <br/>
[paper](https://arxiv.org/pdf/2103.15858.pdf) | [code](https://github.com/lambert-x/CateNorm) | [slides](https://www.zongweiz.com/_files/ugd/deaea1_c35eabb3d59f49ecbd39c42efd551f1c.pdf)


## Install/Check dependencies:
   ```shell
    pip install requirements.txt
   ```
## Prepare Dataset
### 1. Download dataset
1. Prostate: We use the preprocessed [multi-site dataset for prostate MRI segmentation.](https://liuquande.github.io/SAML/)
2. Abdominal: We use the [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) datasets. 
For single domain(site) experiments, we directly use BTCV with the official annotations.
For multiple domain(site) experiments, we use the annotation published in [here.](https://zenodo.org/record/1169361#.YFqGYK_0lm_)

### 2. Preprocess data
1. Prostate: Datasets have already been preprocessed.
2. Abdominal: There are two jupyter notebooks in `./preprocess` for preprocessing data with different settings.
### 3. Generate text file for each site
Each site-wise folder needs a text file(all_list.txt) including paths of all cases.
`data_list_generator.ipynb` is offered to help you generate.

### 4. Overview of a dataset folder
A dataset folder should look like this:

    Dataset/Prostate_Multi/
    ├── Site-A
    │   ├── all_list.txt
    │   ├── Case00.nii.gz
    │   ├── Case00_segmentation.nii.gz
    │   ├── Case01.nii.gz
    │   ├── Case01_segmentation.nii.gz
    │   ...
    │
    ├── Site-B
    │   ├── all_list.txt
    │   ├── Case00.nii.gz
    │   ├── Case00_segmentation.nii.gz
    │   ├── Case01.nii.gz
    │   ├── Case01_segmentation.nii.gz
    │   ...
    │
    ├── Site-C
    │   ├── all_list.txt
    │   ├── Case00.nii.gz
    │   ├── Case00_segmentation.nii.gz
    │   ├── Case01.nii.gz
    │   ├── Case01_segmentation.nii.gz
    │   ...
    │
### 5. Set up the data and result paths 
Please modify `"data_dir"` and `"save_dir"` in `train.py` & `test.py` with your own configuration.
   ```shell
    data_dir = {'local-prostate': 'G:/Dataset/Prostate_Multi_Site',
                'local-ABD-8': 'G:/Dataset/Abdominal_Single_Site_8organs',
                'local-ABD-6': 'G:/Dataset/Abdominal_Multi_Site_6organs',
                }

    save_dir = {'local-prostate': 'G:/DualNorm-Unet/',
                'local-ABD-8': 'G:/DualNorm-Unet/',
                'local-ABD-6': 'G:/DualNorm-Unet/',
                }
   ```

## Training
1. Baseline (single site): 
    
   ```shell
   # Here we use prostate site A as an example
   
   python train.py  --save-fold=Prostate-Single-A --batch-size=4 --sitename A --epochs=20
   ```
2. Baseline (Multiple site): 
    
   ```shell
   # Here we use prostate sites A,B,C as an example
   
   python train.py  --save-fold=Prostate-Multi-ABC --batch-size=6 --sitename ABC --epochs=20
   ```
3. Load Pretraining Model
   ```shell
   # Here we use prostate sites A,B,C as an example
   
   python train.py  --save-fold=Prostate-Multi-ABC-Pretrained --batch-size=6 --sitename ABC --epochs=10 \
   --load=G:/DualNorm-Unet/checkpoints/xxx/xxx/Epochs_10_Aug_True_Zoom_False_Nonlinear_relu_Norm_BN
   ```
3. DualNorm-Unet : 
    
   ```shell
   # Here we use prostate sites A,B,C with DualNorm blocks 1-4(inc, down1, down2, down3) as an example
   
   python train.py  --save-fold=Prostate-Multi-ABC-Pretrained --batch-size=6 --sitename ABC --epochs=10 \
   --load=G:/DualNorm-Unet/checkpoints/xxx/xxx/Epochs_10_Aug_True_Zoom_False_Nonlinear_relu_Norm_BN \
   --spade-aux-blocks inc down1 down2 down3
   ```
   
## Testing

To evaluate the model and save predictions, run:
   ```shell
   python test.py  --save-fold=Prostate-Multi-ABC-Test --batch-size=6 --sitename ABC \
   --load=G:/DualNorm-Unet/checkpoints/xxx/xxx/Epochs_10_Aug_True_Zoom_False_Nonlinear_relu_Norm_BN \
   --save-prediction=True
   ```
   All the predictions are saved as `.nii` files in the `prediction_nii` folder e.g., `G:/DualNorm-Unet/prediction_nii`.
   
   
## Reference:
- https://github.com/milesial/Pytorch-UNet
- https://github.com/liuquande/MS-Net
 
## Citations

```bibtex
@inproceedings{xiao2022catenorm,
  title={CateNorm: Categorical Normalization for Robust Medical Image Segmentation},
  author={Xiao, Junfei and Yu, Lequan and Zhou, Zongwei and Bai, Yutong and Xing, Lei and Yuille, Alan and Zhou, Yuyin},
  booktitle={MICCAI Workshop on Domain Adaptation and Representation Transfer},
  pages={129--146},
  year={2022},
  organization={Springer}
}
```
