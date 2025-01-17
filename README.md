# Age Progression and Facial Alteration System for Forensic Investigations using GANs

**Siddharth Kamath, Varun S Amin, Vinay Kumar U**

We present a GAN-powered framework designed to aid forensic investigations by addressing critical challenges in image processing. Our approach integrates advanced deep generative models such as HRFAE and Pixel2Style2Pixel (pSp) to perform two key tasks: age progression of facial images and transformation of sketches into photo-realistic images. Leveraging the capabilities of Generative Adversarial Networks (GANs) and the multi-faceted CelebA dataset, this framework enhances image synthesis for forensic applications, offering tools for improved identity verification and reconstruction of inferior-quality imagery. These advancements significantly contribute to the accuracy and visual presentation of forensic evidence for law enforcement.

**Key Features:**

* **Enhanced Forensic Investigations:** Provides valuable tools for facial reconstruction, suspect identification, and missing person searches.
* **State-of-the-art GANs:** Utilizes the advanced pSp GAN model for high-quality image generation.
* **Accurate Age Progression:** Incorporates HRFAE aging techniques to simulate realistic aging effects.
* **Potential for Case Verification:** Helps law enforcement in verifying evidence and identifying potential suspects.

**Project Goals:**

* Develop a robust and accurate system for generating realistic facial images from sketches.
* Implement effective age progression techniques to simulate aging across different age ranges.
* Evaluate the system's performance on real-world forensic case scenarios.
* Explore potential applications beyond traditional forensic investigations.

## Process Flow
<div id="header" align="center">
  <img src="/assets/Picture1.png" width="80%" /> 
</div>

---

## Requirements and Installations

Before running the project, ensure you have the required dependencies installed. Below is a list of the core libraries and PyTorch dependencies with their respective versions.

### Core Libraries
The following Python packages are required:
- `numpy==1.18.4`
- `scipy==1.4.1`
- `matplotlib==3.2.1`
- `tqdm==4.46.0`
- `opencv-python==4.2.0.34`
- `pillow==7.1.2`
- `pyyaml==5.3.1`
- `protobuf==3.11.4`
- `tensorboard==2.2.1`
- `tensorboard-logger==0.1.0`
- `tensorboardx==2.0`

### PyTorch and CUDA
Ensure you install the specific version of PyTorch and torchvision. This project requires the following versions:
- `torch==1.6.0`
- `torchvision==0.4.2`

If you need a specific PyTorch version compatible with CUDA 10.0, use the following:
- `torch==1.1.0+cu100`
- `torchvision==0.3.0+cu100`

You can install these packages using the following command:

```bash
pip install -r requirements.txt
```

---

## Technologies Used
- Python
- TensorFlow
- PyTorch
- GAN-based architectures


---

## **Usage**

### **Step 1: Clone Repositories**

#### **Sketch-to-Face Conversion**
Clone the repository for sketch-to-face transformation:
```bash
git clone https://github.com/AkhileshV/Sketch-to-face.git
```

#### **Age Progression**
Clone the repository for age progression using the HRFAE model:
```bash
git clone https://github.com/InterDigitalInc/HRFAE.git
```

#### **Sketch Simplification Tool**
For sketch cleanup and simplification, clone the following repository. It includes pre-trained models from the paper _"Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup"_:
```bash
git clone https://github.com/bobbens/sketch_simplification.git
```

### **Step 2: Follow Module Instructions**
Each repository contains detailed instructions for setup and execution. Follow these instructions after cloning to ensure proper functionality.

---


## Visual Gallery
Here we showcase sample images demonstrating the capabilities of our framework across various tasks:

**Image to Sketch Conversion (for Dataset Creation):**
The transformation of realistic images into simplified sketches to prepare a diverse dataset for training and testing.
<div id="header" align="center">
  <img src="/assets/F2S.png" width="80%"/> 
</div>
<br>

**Sketch to Image Conversion:**
Translating basic sketches into photo-realistic images, preserving structural integrity and fine details.
<div id="header" align="center">
  <img src="/assets/S2F.png" width="80%"/> 
</div>
<br>

**Age Progression:**
Generating realistic age-progressed versions of facial images to illustrate identity evolution over time.
<div id="header" align="center">
  <img src="/assets/AP.png" width="100%"/> 
</div>
<br>
