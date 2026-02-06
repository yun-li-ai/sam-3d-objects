# Agentic 2D→3D Asset Generation System

A modular AI-powered system that converts 2D images into high-quality 3D assets using multiple specialized agents.

## 🚀 Quick Start

### 1. Create Conda Environment

**Option A: Using environment.yml (Recommended)**
```bash
cd agentic
conda env create -f environment.yml
conda activate agentic
```

**Option B: Manual Setup**
```bash
# Create new environment
conda create -n agentic python=3.11 -y
conda activate agentic

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r ../requirements.inference.txt  # For SAM-3D-Objects
```

### 3. Download Models (First Time Only)

```bash
python download_models.py
```

This will cache the SAM-3D models locally (~2GB).

### 4. Set Up API Key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Get your API key from: https://makersuite.google.com/app/apikey

### 5. Run the Application

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## 📁 Project Structure

```
agentic/
├── backend/                    # Core backend modules
│   ├── agents/                 # AI agents (segmentation, generation, 3D)
│   │   ├── gemini_segmentation_agent.py
│   │   ├── image_generation_agent.py
│   │   └── generation_3d_agent.py
│   ├── services/              # External services (Gemini, SAM-3D, Storage)
│   │   ├── gemini_service.py
│   │   ├── sam3d_service.py
│   │   └── storage_service.py
│   ├── utils/                 # Utilities (mask parsing)
│   ├── models.py              # Data models
│   └── orchestrator.py        # Main coordinator
├── config/
│   └── agentic_system.yaml    # Configuration
├── data/jobs/                 # Persistent storage for all jobs
├── app.py                     # Web interface
├── download_models.py         # Model download script
└── requirements.txt           # Dependencies
```

## 🎯 Features

- 🤖 **Multi-agent architecture** - Specialized agents for each task
- 🖼️ **Google Gemini integration** - Object segmentation and image generation
- 🎨 **Style editing** - Edit objects with text prompts
- 🏗️ **SAM-3D-Objects** - High-quality 3D asset generation
- 🔄 **Queue-based 3D generation** - Handles GPU constraints
- 💾 **Job persistence** - All data saved locally, resume after restart
- 📦 **Batch processing** - Generate multiple objects at once
- 🌐 **Interactive web UI** - Row-based layout with real-time updates

## 💻 Usage

### Web Interface

1. **Upload Image** - Upload any 2D image
2. **Segment Objects** - System detects and segments objects automatically
3. **Review & Edit** - Each row shows:
   - Masked original (object highlighted in context)
   - Generated clean image (transparent background)
   - Edit prompt (optional style changes)
   - Generate 3D button
4. **Generate 3D Assets** - Click to generate 3D models (PLY + GLB)
5. **View Results** - Interactive 3D viewer in browser

### Python API

```python
from backend.orchestrator import AgenticOrchestrator
from PIL import Image

# Initialize
orch = AgenticOrchestrator()

# Create job and segment
image = Image.open("your_image.jpg")
job = orch.create_job_from_image(image)
job, overlay = orch.segment_image(job)
orch.generate_clean_images(job)

# Generate 3D
object_ids = [obj.object_id for obj in job.objects]
orch.submit_3d_generation(job, object_ids)

while not orch.generation_queue.is_empty():
    orch.process_3d_queue(max_iterations=1)
```

## 🏗️ Architecture

### Workflow

```
Upload → Segment → Generate Clean Images → Edit (optional) → Generate 3D → Download
```

### Components

- **Orchestrator**: Coordinates all agents and services
- **Agents**: Specialized AI agents (segmentation, generation, 3D)
- **Services**: External API wrappers (Gemini, SAM-3D, Storage)
- **Models**: Data structures (Job, SegmentedObject, Asset3D, Queue)

## 📊 Output Structure

All data is stored locally in `data/jobs/{job_id}/`:

```
data/jobs/{job_id}/
├── original.png                # Uploaded image
├── image_resized.png           # Resized for processing (1024px max)
├── overlay_masks.png           # All objects highlighted
├── job_metadata.json           # Job data
└── objects/
    └── obj_{n}/
        ├── mask.png            # Segmentation mask
        ├── masked_original.png # Object in context
        ├── generated.png       # Clean image (transparent bg)
        ├── edited.png          # Edited (if edited)
        └── assets/
            ├── model.glb       # 3D mesh
            └── model.ply       # Gaussian splat
```

## 🔧 Configuration

Edit `config/agentic_system.yaml` to customize:

```yaml
gemini:
  segmentation_model: "gemini-2.5-pro"
  image_model: "gemini-3-pro-image-preview"

sam3d:
  config_path: "../checkpoints/hf/pipeline.yaml"
```

## 🐛 Troubleshooting

### API Key Issues
```bash
echo $GEMINI_API_KEY  # Check if set
export GEMINI_API_KEY="your-key"
```

### Model Download Issues
Models are cached in `~/.cache/huggingface/`. First run will download ~2GB.

### GPU Memory
- Process fewer objects at once
- Monitor with `nvidia-smi`
- Models require 8GB+ VRAM

## 📦 Requirements

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Google Gemini API key

## 📄 License

See parent directory LICENSE file.

---

**Built with**: Google Gemini API, SAM-3D-Objects, Gradio
