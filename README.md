# AdaIN Neural Style Transfer 🎨

An image stylization project built using **PyTorch, Flask, and Adaptive Instance Normalization (AdaIN)**. The application allows users to upload a content image and a style image, adjust the style strength, and generate a stylized result through a web interface.

The repository includes a Flask application, a VGG-based feature encoder, a trainable decoder, a training script, and a notebook for visualizing image features.

## ✨ Features

- **Image-to-image style transfer:** Combine the content of one image with the visual style of another.
- **AdaIN-based stylization:** Align the channel-wise feature statistics of content and style images.
- **Interactive web interface:** Upload images through a Flask application.
- **Adjustable style strength:** Control the stylization using an alpha slider from `0.0` to `1.0`.
- **Image previews:** Preview uploaded content and style images.
- **Downloadable output:** Save the generated stylized image.
- **GPU support:** Uses CUDA when available and otherwise falls back to CPU.
- **Custom decoder training:** Train a decoder using content and style datasets.
- **Feature visualization:** Inspect VGG feature activations using a Jupyter notebook.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Neural network architecture and inference |
| Torchvision | Image transformations and tensor utilities |
| VGG encoder | Extract image feature representations |
| AdaIN | Transfer style statistics to content features |
| Pillow | Image loading and saving |
| Flask | Web application |
| HTML, CSS, JavaScript | User interface and image previews |
| Matplotlib | Feature visualization |
| Gunicorn | WSGI server for deployment |
| tqdm | Training progress display |

## 🧠 How It Works

The project uses an AdaIN-based neural style transfer pipeline.

```text
Content Image                 Style Image
      |                             |
      v                             v
 VGG Encoder                   VGG Encoder
      |                             |
      v                             v
 Content Features             Style Features
      |                             |
      +-------------+---------------+
                    |
                    v
          Adaptive Instance
             Normalization
                    |
                    v
            Blended Features
              using alpha
                    |
                    v
                 Decoder
                    |
                    v
            Stylized Output Image
```

### Main Components

**1. VGG Encoder**

Extracts hierarchical image features using convolutional layers. Its pretrained weights are loaded from `vgg_normalised.pth`, and its parameters are frozen during decoder training.

**2. Adaptive Instance Normalization (AdaIN)**

Normalizes content features and adjusts their channel-wise mean and standard deviation to match the style features.

**3. Decoder**

Reconstructs an RGB image from the transformed feature representation using convolutional layers and nearest-neighbor upsampling.

**4. Alpha blending**

The inference pipeline blends the transformed features with the original content features:

```python
t = alpha * t + (1 - alpha) * c_feats
```

- `alpha = 0.0`: Uses the original content features before decoding.
- `alpha = 1.0`: Uses the fully AdaIN-transformed features.
- Intermediate values blend the two representations.

The final visual result also depends on the trained decoder and input images.

## 📁 Project Structure

```text
nst_dl/
├── NST_Code/
│   ├── app.py
│   ├── train.py
│   ├── requirements.txt
│   ├── vgg_normalised.pth
│   ├── adain_algo.png
│   ├── content_data/
│   ├── style_data/
│   ├── examples/
│   ├── experiment1/
│   ├── experiment/
│   │   └── final_exp/
│   │       └── decoder_final.pth
│   ├── static/
│   │   └── uploads/
│   ├── templates/
│   │   └── index.html
│   └── utils/
│       ├── models.py
│       └── utils.py
├── code.ipynb
├── requirements.txt
├── Procfile.txt
├── .gitignore
└── README.md
```

The structure above highlights the main components. The exact contents of experiment folders and generated artifacts may vary.

## ⚙️ Installation and Setup

### Prerequisites

- Python compatible with the pinned PyTorch and Torchvision versions
- pip
- Git
- A trained decoder checkpoint and pretrained VGG weights
- CUDA-compatible hardware and drivers for GPU acceleration (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/2k24csaiml1b2414254-lgtm/nst_dl.git
cd nst_dl
```

### 2. Install Dependencies

The application and training script are located inside `NST_Code/`.

```bash
cd NST_Code
pip install -r requirements.txt
```

The project pins PyTorch `2.2.2` and Torchvision `0.17.2`. Check compatibility with your Python version and hardware before installation.

For a GPU setup, install the appropriate PyTorch build for your CUDA environment if the pinned installation does not provide the required CUDA support.

### 3. Verify Model Files

The Flask application expects these files:

```text
NST_Code/
├── vgg_normalised.pth
└── experiment/
    └── final_exp/
        └── decoder_final.pth
```

The VGG weights and decoder checkpoint must exist at these locations before launching the application. The repository's training script does not automatically create the specific `decoder_final.pth` filename expected by the application.

## 🚀 Run the Web Application

From inside `NST_Code/`, run:

```bash
python app.py
```

Open the local address printed by Flask in your browser.

### Using the Application

1. Upload a **content image**.
2. Upload a **style image**.
3. Adjust the **Style Strength** slider.
4. Click **Generate Styled Image**.
5. Preview the result and download the generated image.

Supported filename extensions are PNG, JPG, and JPEG. The application resizes and center-crops input images to `256 × 256` before inference.

## 🏋️ Train the Decoder

The training script accepts arguments for dataset directories, model weights, experiment name, image sizes, batch size, learning rate, and training duration.

Example command from inside `NST_Code/`:

```bash
python train.py \
  --content_dir content_data \
  --style_dir style_data \
  --vgg vgg_normalised.pth \
  --experiment experiment1 \
  --epochs 2
```

This example uses relative paths and assumes the datasets and VGG weights are available in the specified locations.

### Training Configuration

The script defines these default settings:

| Parameter | Default |
|---|---:|
| Final image size | 256 |
| Content image size | 512 |
| Style image size | 512 |
| Batch size | 4 |
| Learning rate | 0.0001 |
| Learning-rate decay | 0.00005 |
| Epochs | 1 |
| Content loss weight | 1.0 |
| Style loss weight | 5 |
| Log interval | 1 epoch |
| Save interval | 2 epochs |

The encoder is kept frozen while the decoder is optimized using content and style losses.

### Training Outputs

The script creates an experiment directory containing training arguments. At configured save intervals, it saves decoder checkpoints, optimizer states, and example output images.

**Important:** The default number of epochs is `1` and the default save interval is `2`. With those defaults, the checkpoint-saving condition is not reached. Increase the number of epochs or change the save interval when training a checkpoint.

The web application expects its decoder checkpoint at `experiment/final_exp/decoder_final.pth`; configure your training and checkpoint placement accordingly.

## 📊 Feature Visualization Notebook

The root-level `code.ipynb` notebook explores VGG feature activations for an original image and stylized image examples.

It attempts to:

- Load example images.
- Extract features from selected VGG layers.
- Normalize activation maps.
- Visualize feature maps alongside the input images.

Run the notebook in Jupyter after installing the relevant dependencies and verifying the notebook's model imports, weight path, and image paths.

## 🌐 Deployment

The repository includes a Gunicorn startup command in `Procfile.txt`:

```text
web: gunicorn --bind :$PORT app:app
```

This command is intended to start the Flask application using Gunicorn on the port supplied by the hosting platform.

Before deploying:

1. Confirm the platform's expected Procfile name and format.
2. Ensure the working directory contains `app.py` and the required `utils/` package.
3. Verify that both model files are available.
4. Ensure the host has enough memory for PyTorch and model inference.
5. Disable Flask debug mode for production.
6. Configure writable storage for uploaded and generated images.

The repository's startup command alone does not confirm that a deployment is live or successful.

## 🔍 Current Limitations and Considerations

- The Flask application requires both pretrained VGG weights and a compatible trained decoder checkpoint.
- The training script's default dataset and VGG paths are machine-specific; pass suitable paths for your environment.
- The notebook may require corrections to imports, image paths, and visualization loops before it runs successfully.
- Upload validation checks file extensions but does not fully validate image content.
- Uploaded and generated files are stored under `static/uploads/`; a production deployment should consider cleanup, storage limits, and file handling.
- The Flask app currently uses `debug=True` when launched directly; this should be disabled for production.
- No benchmark scores or inference-speed claims are included because verified measurements have not been provided.

## 🔮 Potential Improvements

- Add stronger image validation and upload-size limits.
- Add automated tests for image preprocessing, inference, and output generation.
- Improve checkpoint management and training configuration.
- Add sample outputs generated by the application.
- Add deployment-specific configuration and production logging.
- Document verified training results and hardware performance.

## 👨‍💻 Author

**Himesh Mishra**

GitHub: [2k24csaiml1b2414254-lgtm](https://github.com/2k24csaiml1b2414254-lgtm)

---

*An implementation project exploring AdaIN-based Neural Style Transfer, PyTorch model training, and deployment through a Flask web application.*
