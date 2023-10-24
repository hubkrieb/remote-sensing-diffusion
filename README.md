# Diffusion Model based Data Augmentation for Remote Sensing Imagery
Master Thesis of Hubert Kriebitzsch at the TU Berlin Faculty IV Computer Vision and Remote Sensing Department

## Abstract
Data augmentation is a crucial challenge in deep learning and especially in remote sensing where data is often more difficult and costly to acquire especially when collecting data of rare events such as natural disasters. Many solutions have been proposed to this problem and data augmentation using synthetic data, mainly generated using Generative Adversarial Networks, is one of the most recent and efficient approaches to counter the effects of class imbalance. In this thesis, we further study data augmentation with synthetic data using state-of-the-art generative models. We use diffusion models to generate new remote sensing images for data augmentation purposes. To generate high-fidelity satellite images of active fire, we finetune the foundation model Stable Diffusion using Dreambooth and existing wildfire images. We apply it to the task of active fire detection by inpainting synthetic wildfires into existing satellite images. This allows us to augment semantic segmentation datasets and not only image classification datasets. We conduct a series of experiments to measure the efficiency of the methods proposed and compare different pretrained and finetuned diffusion models as well as different inpainting masks. We evaluate this approach on a small manually annotated active fire detection dataset and achieve an improvement of the dice coefficient from 58.5% up to 72.7%. This work provides new insights on remote sensing data generation with diffusion models, as well as the efficiency of data augmentation using synthetic data generated with them. It presents a novel way to generate semantic segmentation data in remote sensing.

Add figure intro

## Dataset

We use the manually annotated dataset of active fire detection created by de Almeida Pereira et al. available [here](https://github.com/pereira-gha/activefire). It consists of 13 Landsat-8 images captured in September 2020 from different location around the world. They were split into 9,044 256x256 pixels non-overlapping patches. To simplify the image synthesis, we restrict the data to the RGB visible bands.

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## Finetuning Stable Diffusion Inpainting with Dreambooth

To finetune Stable Diffusion Inpainting you can use the inpainting_fintenuning.ipynb file, preferably run it using Google Colab. It uses the following [repository](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/dreambooth_inpaint) to finetune the model. We use Google Drive to store training images and model checkpoints and connect the notebook to it. 

<a target="_blank" href="https://colab.research.google.com/github/hubkrieb/remote-sensing-diffusion/blob/master/data_augmentation/inpainting_finetuning.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Generating Synthetic Data

To generate synthetic data we use a modified version of Automatic1111 Stable Diffusion WebUI that allows to generate a synthetic image for each pair of background and mask in batch mode. 

## Data Augmentation

Provided synthetic images and the corresponding masks in tif format (you can use png_to_tif.py to convert them), you can generate an augmented dataset using augment.py.

```bash
python data_augmentation/augment.py \
--input_path path_to_synthetic_images \
--output_path path_to_augmented_dataset_landsat_patches \
--mask_path path_to_inpainting_masks \
--output_mask_path path_to_augmented_dataset_masks \
--original_image_path path_to_original_dataset_landsat_patches \
--original_mask_path path_to_original_dataset_manual_annotations_patches \
--split_path path_to_original_dataset_split \
--output_split_path path_to_augmented_dataset_split \
--lmdb_path path_to_lmdb
```

## Training Semantic Segmentation U-Net

Once you have a ready-to-use LMDB database, you can train the U-Net model.

```bash
python semantic_segmentation/run.py \
--epochs 200 \
--batch_size 8 \
--n_channels 3 \
--checkpoint_interval 10 \
--checkpoint_path path_to_checkpoints \
--lmdb_path path_to_lmdb_dir \
--split_path path_to_split
```