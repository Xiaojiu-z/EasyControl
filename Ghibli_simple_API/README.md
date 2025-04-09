# Ghibli Style Image Generation API

This project provides a simple API for generating images in the style of Studio Ghibli animations. It uses a pre-trained FLUX model with EasyControl to achieve the distinctive Ghibli art style.

## Features

- Generate Ghibli-style images from input images
- RESTful API interface
- Health check endpoint
- Configurable generation parameters
- Support for various image sizes (256x256 to 1024x1024)

## Requirements

- Python 3.7+
- FastAPI
- PyTorch
- PIL (Python Imaging Library)
- requests
- requests_toolbelt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Xiaojiu-z/EasyControl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained models from the space:
```bash
wget -P models/ https://huggingface.co/spaces/jamesliu1217/EasyControl_Ghibli/resolve/main/models/Ghibli.safetensors

```

## Usage

### Starting the Server

Run the FastAPI server:
```bash
cd Ghibli_simple_API
python service.py
```

The server will start at `http://your_ip:7860`

### Health Check

Check if the service is running:
```bash
python health.py
```

Expected output:
```
[✅] Service online (Check time: YYYY-MM-DD HH:MM:SS)
    - Status: HEALTHY
    - Model loaded: Success
```

### Generating Images

Use the client script to generate images:
```bash
python client.py
```

#### Customizing Generation Parameters

You can customize the generation parameters by passing them as keyword arguments to the `send_generation_request` function:

```python
from client import send_generation_request

# Example with custom parameters
send_generation_request(
    "input_image.png",
    "custom_output.png",
    num_inference_steps=13,     # Less steps for quicker response
    guidance_scale=3.5,
    seed=123,                   # Different seed for variation
    height=1024,
    width=768,                  # Adjust width and height accordingly.
)
```

Available parameters:
- `prompt`: Text description for image generation
- `height`: Image height (256-1024)
- `width`: Image width (256-1024)
- `seed`: Random seed for reproducibility
- `guidance_scale`: Classifier-free guidance scale (typically 1.0-10.0)
- `num_inference_steps`: Number of denoising steps (higher = better quality but slower)
- `max_sequence_length`: Maximum token sequence length

### API Endpoints

#### POST /generate
Generate a Ghibli-style image from an input image.

**Request Parameters:**
- `spatial_image`: Input image file
- `request`: JSON payload with generation parameters:
  ```json
  {
    "prompt": "Ghibli Studio style, Charming hand-drawn anime-style illustration",
    "height": 768,
    "width": 768,
    "seed": 42,
    "control_type": "Ghibli",
    "guidance_scale": 3.5,
    "num_inference_steps": 25,
    "max_sequence_length": 512
  }
  ```

**Response:**
- Success: PNG image file
- Error: JSON with error details

#### GET /health
Check the health status of the service.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Troubleshooting

### Parameters Not Taking Effect

If you find that changing parameters in the client does not affect the generation results, make sure that:

1. The client is correctly passing parameters to the API request
2. The server is correctly parsing and using these parameters
3. The parameter values are within valid ranges

The original client implementation had a limitation where parameters set in the code were not applied to the actual request. This has been fixed by adding the ability to pass parameters as keyword arguments.

## Configuration

The following parameters can be adjusted in the generation request:

- `prompt`: Text description for image generation
- `height`: Image height (256-1024)
- `width`: Image width (256-1024)
- `seed`: Random seed for reproducibility
- `guidance_scale`: Classifier-free guidance scale (e.g. 3.5)
- `num_inference_steps`: Number of denoising steps (e.g. 25)
- `max_sequence_length`: Maximum token sequence length (e.g. 512)

## Error Handling

The API handles various error cases:
- Network errors
- Invalid JSON responses
- Parameter validation errors
- Image processing errors