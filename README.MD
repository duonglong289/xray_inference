# X-Ray Inference

This repository contains code for X-Ray inference.

## Setup and Usage

Follow these steps to set up and run the X-Ray inference server:

### 1. Installation

Install the required dependencies:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

### 2. Download Weights

Download the model weights from the following link:

[script_resnet.ts](https://drive.google.com/file/d/1GgsxtQJVKxspC54XMnUbeW-E8o_ZR1Jq/view?usp=sharing)

[model_faster_rcnn.ts](https://drive.google.com/file/d/1Pb_6_HSHn5oLq5jBwgFx9y8_FRsNDhxE/view?usp=drive_link)

Place the downloaded weights in the same directory as the script.

### 3. Run Server

Start the inference server by running:

```bash
python app.py
```

The server will start and be ready to process X-Ray images for inference.