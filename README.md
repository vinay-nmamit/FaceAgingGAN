# Age Progression and Facial Alteration System for Forensic Investigations using GANs

This repository contains the code and models for an innovative system that leverages Generative Adversarial Networks (GANs) to assist in forensic investigations. The system focuses on two key aspects:

1. **Sketch-to-Image Generation:** It transforms incomplete sketches into lifelike images using the powerful Pixel2Style2Pixel (pSp) GAN model. This enables investigators to generate more accurate visualizations of potential suspects or missing persons based on limited eyewitness descriptions or artistic renditions.

2. **Age Progression:** The system incorporates HRFAE aging technique to simulate the natural aging process on the generated images. This allows investigators to predict how a person's appearance might change over time, aiding in the identification of individuals who may have aged since the initial incident.

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

## Installation

Before running the project, ensure you have the required dependencies installed. Below is a list of the core libraries and PyTorch dependencies with their respective versions.

### Core Libraries
The following Python packages are required:
- `numpy==1.18.1`
- `scipy==1.1.0`
- `pillow==6.1.0`
- `pyyaml==5.3.1`
- `protobuf==3.11.4`
- `tensorboard-logger==0.1.0`
- `tensorboardx==2.0`

### PyTorch and CUDA
Ensure you install the specific version of PyTorch and torchvision compatible with CUDA 10.0:
- `torch==1.1.0+cu100`
- `torchvision==0.3.0+cu100`

You can install these packages using the following command:

```bash```
pip install -r requirements-HRFAE.txt requirements-PSP.txt

## Technologies Used
- Python
- TensorFlow
- PyTorch
- GAN-based architectures

## Usage
Clone the repository using 
```bash
git clone https://github.com/vinay-nmamit/FaceAgingGAN.git
```
