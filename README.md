# IP-Adapter FaceID Image Generator

A FastAPI server that uses **h94/IP-Adapter-FaceID** and **SG161222/Realistic_Vision_V4.0_noVAE** to generate consistent pictures of people based on face ID embeddings.

## Features

-   **Multiple IP-Adapter variants**: FaceID, FaceID-Plus, FaceID-PlusV2, and Portrait modes
-   **Consistent face generation**: Uses face ID embeddings for identity preservation
-   **RESTful API**: Easy-to-use FastAPI endpoints
-   **Multiple generation modes**:
    -   Standard: Single face image input
    -   Plus: Enhanced with face structure control
    -   Portrait: Multiple face images for better consistency
-   **Customizable parameters**: Control image dimensions, inference steps, guidance scale, etc.

## Requirements

-   Python 3.8+
-   CUDA-compatible GPU (recommended) or CPU
-   8GB+ VRAM for optimal performance
-   20GB+ free disk space for models

## Installation

### 1. Clone this repository

```bash
git clone <repository-url>
cd image-gen-faceid
```

### 2. Run the setup script

```bash
python setup.py
```

This will:

-   Install all required Python packages
-   Clone and install the IP-Adapter repository
-   Download the required model files

### 3. Alternative manual installation

If the setup script fails, you can install manually:

```bash
# Install requirements
pip install -r requirements.txt

# Clone IP-Adapter
git clone https://github.com/tencent-ailab/IP-Adapter.git
cd IP-Adapter
pip install -e .
cd ..

# Create models directory
mkdir models
```

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`. You can view the API documentation at `http://localhost:8000/docs`.

### API Endpoints

#### 1. Standard Generation (`/generate`)

Generate images using basic IP-Adapter-FaceID:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "face_image=@path/to/face.jpg" \
  -F "prompt=photo of a woman in red dress in a garden" \
  -F "num_samples=2" \
  -F "width=512" \
  -F "height=768"
```

#### 2. Plus Generation (`/generate-plus`)

Generate images with enhanced face structure control:

```bash
curl -X POST "http://localhost:8000/generate-plus" \
  -F "face_image=@path/to/face.jpg" \
  -F "prompt=portrait of a person in professional attire" \
  -F "s_scale=1.2" \
  -F "shortcut=true" \
  -F "num_samples=1"
```

#### 3. Portrait Generation (`/generate-portrait`)

Generate images using multiple reference face images:

```bash
curl -X POST "http://localhost:8000/generate-portrait" \
  -F "face_images=@face1.jpg" \
  -F "face_images=@face2.jpg" \
  -F "face_images=@face3.jpg" \
  -F "prompt=professional headshot" \
  -F "width=512" \
  -F "height=512"
```

### Parameters

| Parameter             | Type    | Default                 | Description                           |
| --------------------- | ------- | ----------------------- | ------------------------------------- |
| `face_image`          | File    | Required                | Input face image (single image)       |
| `face_images`         | File[]  | Required                | Input face images (for portrait mode) |
| `prompt`              | String  | Required                | Text description for image generation |
| `negative_prompt`     | String  | "monochrome, lowres..." | What to avoid in generation           |
| `num_samples`         | Integer | 1                       | Number of images to generate (1-4)    |
| `width`               | Integer | 512                     | Image width (256-1024)                |
| `height`              | Integer | 768                     | Image height (256-1024)               |
| `num_inference_steps` | Integer | 30                      | Denoising steps (10-100)              |
| `guidance_scale`      | Float   | 7.5                     | Guidance scale (1.0-20.0)             |
| `seed`                | Integer | None                    | Random seed for reproducibility       |
| `s_scale`             | Float   | 1.0                     | Face structure scale (Plus mode only) |
| `shortcut`            | Boolean | False                   | Use v2 shortcut (Plus mode only)      |

### Response Format

All endpoints return JSON with the following structure:

```json
{
	"success": true,
	"images": ["base64_encoded_image_1", "base64_encoded_image_2"],
	"message": "Successfully generated 2 images"
}
```

## Python Client Example

```python
import requests
import base64
from PIL import Image
import io

def generate_image(face_image_path, prompt):
    url = "http://localhost:8000/generate"

    with open(face_image_path, "rb") as f:
        files = {"face_image": f}
        data = {
            "prompt": prompt,
            "num_samples": 1,
            "width": 512,
            "height": 768
        }

        response = requests.post(url, files=files, data=data)
        result = response.json()

        if result["success"]:
            # Decode first image
            image_data = base64.b64decode(result["images"][0])
            image = Image.open(io.BytesIO(image_data))
            image.save("generated_image.png")
            print("Image saved as generated_image.png")
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")

# Usage
generate_image("my_face.jpg", "photo of a person in a business suit")
```

## Model Information

### Base Models Used

-   **Stable Diffusion**: SG161222/Realistic_Vision_V4.0_noVAE
-   **VAE**: stabilityai/sd-vae-ft-mse
-   **Face Recognition**: InsightFace buffalo_l

### IP-Adapter Variants

1. **FaceID**: Basic face identity preservation
2. **FaceID-Plus**: Face ID + CLIP image embedding for structure
3. **FaceID-PlusV2**: Controllable face structure weight
4. **FaceID-Portrait**: Multi-image face consistency (no LoRA/ControlNet)

## Performance Tips

1. **GPU Memory**: Reduce `num_samples` or image dimensions if you encounter CUDA OOM errors
2. **Quality vs Speed**: Higher `num_inference_steps` (50-100) for better quality
3. **Consistency**: Use `seed` parameter for reproducible results
4. **Portrait Mode**: Use 3-5 similar face images for best results

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:

    - Reduce image dimensions or num_samples
    - Use CPU by setting device="cpu" in services

2. **No Face Detected**:

    - Ensure face is clearly visible and well-lit
    - Face should be reasonably large in the image

3. **Model Download Failures**:

    - Check internet connection
    - Manually download models to `models/` directory

4. **Import Errors**:
    - Run `python setup.py` again
    - Ensure all requirements are installed

### Logs

Check the console output for detailed error messages and generation progress.

## License

This project is for research purposes only. The IP-Adapter-FaceID models are subject to their respective licenses:

-   IP-Adapter: Non-commercial research use
-   InsightFace: Non-commercial research use
-   Stable Diffusion: CreativeML OpenRAIL-M

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

-   [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) by Tencent AI Lab
-   [InsightFace](https://github.com/deepinsight/insightface) for face recognition
-   [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
