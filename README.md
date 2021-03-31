# DualNorm-UNet: Incorporating Global and Local Statistics for Robust Medical Image Segmentation

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
@article{xiao2021dualnorm,
  title={DualNorm-UNet: Incorporating Global and Local Statistics for Robust Medical Image Segmentation},
  author={Xiao, Junfei and Yu, Lequan and Xing, Lei and Yuille, Alan and Zhou, Yuyin},
  journal={arXiv},
  year={2021}
}
```
