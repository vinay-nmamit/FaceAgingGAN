# Age Progression and Facial Alteration System for Forensic Investigations using GANs

**Siddharth Kamath, Varun S Amin, Vinay Kumar U**

We present a GAN-powered framework designed to aid forensic investigations by addressing critical challenges in image processing. Our approach integrates advanced deep generative models such as HRFAE and Pixel2Style2Pixel (pSp) to perform two key tasks: age progression of facial images and transformation of sketches into photo-realistic images. Leveraging the capabilities of Generative Adversarial Networks (GANs) and the multi-faceted CelebA dataset, this framework enhances image synthesis for forensic applications, offering tools for improved identity verification and reconstruction of inferior-quality imagery. These advancements significantly contribute to the accuracy and visual presentation of forensic evidence for law enforcement.

## Key Features

- **Enhanced Forensic Investigations:** Tools for facial reconstruction, suspect identification, and missing person searches.
- **State-of-the-art GANs:** Utilizes advanced GAN models for high-quality image generation.
- **Accurate Age Progression:** Incorporates HRFAE aging techniques to simulate realistic aging effects.
- **Improved Evidence Analysis:** Assists law enforcement in verifying evidence and identifying potential suspects.

## Project Goals

- Develop a robust and accurate system for generating realistic facial images from sketches.
- Implement effective age progression techniques to simulate aging across different age ranges.
- Evaluate the system's performance on real-world forensic case scenarios.
- Explore potential applications beyond traditional forensic investigations.

## Process Flow

<div id="header" align="center">
  <img src="/assets/Picture1.png" width="80%" />
</div>

---

## Requirements and Installations

Ensure you have the required dependencies installed. Below is a list of core libraries and PyTorch dependencies.

### Core Libraries

```plaintext
numpy==1.18.4
scipy==1.4.1
matplotlib==3.2.1
tqdm==4.46.0
opencv-python==4.2.0.34
pillow==7.1.2
pyyaml==5.3.1
protobuf==3.11.4
tensorboard==2.2.1
tensorboard-logger==0.1.0
tensorboardx==2.0
```

### PyTorch and CUDA

Install the required versions:

```plaintext
torch==1.6.0
torchvision==0.4.2
```

For CUDA 10.0 compatibility:

```plaintext
torch==1.1.0+cu100
torchvision==0.3.0+cu100
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Technologies Used

- Python
- TensorFlow
- PyTorch
- GAN-based architectures

## Visual Gallery

### Image to Sketch Conversion (Dataset Creation)

Transform realistic images into sketches for dataset preparation:
<div id="header" align="center">
  <img src="/assets/F2S.png" width="80%" />
</div>

### Sketch to Image Conversion

Generate photo-realistic images from sketches while preserving fine details:
<div id="header" align="center">
  <img src="/assets/S2F.png" width="80%" />
</div>

### Age Progression

Simulate realistic age progression for identity evolution:
<div id="header" align="center">
  <img src="/assets/AP.png" width="100%" />
</div>

---

## Sketch-to-Face

Leverages GANs to convert sketches into realistic human faces.

- **Paper Link:** [https://arxiv.org/abs/2008.00951](https://arxiv.org/abs/2008.00951)
- **Official Repository:** [https://github.com/eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)

### System Requirements

- **OS:** Linux/Mac OS
- **Software:** Python 3.5+, OpenCV, scikit-learn, numpy

### Hardware Used

- **GPU:** Google Colab, 15 GB GPU (Nvidia Tesla T4)

### CelebAHQ Dataset

- **Dataset Link:** [Kaggle](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
- **Synthetic Sketch Generation:** [Reference Code](GAN-s2f/scripts/pencil_sketch_create_dataset.py)

### Training Command for Sketch-to-Face Encoder

Run the following command to train the model:

```bash
python scripts/train.py \
--dataset_type=celebs_sketch_to_face \
--exp_dir=/path/to/exp/dir \
--checkpoint_path=/path/to/save/checkpoint.pt \
--workers=4 \
--batch_size=4 \
--test_batch_size=4 \
--test_workers=4 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0 \
--w_norm_lambda=0.005 \
--label_nc=1 \
--input_nc=1
```

---

## HRFAE: High Resolution Face Age Editing

Official implementation for paper [High Resolution Face Age Editing](https://arxiv.org/pdf/2005.04410.pdf).

- **Repository:** [https://github.com/vadimkantorov/caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch)

### Pretrained Model

Download the model:

```bash
cd ./logs/001
./download.sh
```

### Test Images

Upload test images to `/test/input` and run:

```bash
python test.py --config 001 --target_age 65
```

### Train a New Model

1. **Prepare Dataset:** Download [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset and unzip it to the `/data/ffhq` directory.
2. **Download Age Labels:** Get the age labels from [here](https://partage.imt.fr/index.php/s/DbSk4HzFkeCYXDt) and place them in the `/data` directory.
3. **Custom Dataset:** If you want to train the model with your own dataset, place your images in the `/data` directory. Use the pretrained classifier to generate a label file with the age of each image.
4. **Train Model:** Use the following command to start training:

```bash
python train.py --config 001
```

---

## References and Citations

- **Pixel2Style2Pixel:** Rich, E., et al. "Pixel2Style2Pixel: Encoding in Style the Pixelwise Latent Structure." [Paper Link](https://arxiv.org/abs/2008.00951) | [Repository](https://github.com/eladrich/pixel2style2pixel)
- **HRFAE:** Zhu, H., et al. "High Resolution Face Age Editing." [Paper Link](https://arxiv.org/pdf/2005.04410.pdf) | [Repository](https://github.com/vadimkantorov/caffemodel2pytorch)

