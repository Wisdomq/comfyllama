"""
ComfyUI Dataset Creator
Helper script to create training examples for ComfyUI workflow generation
"""

import json
import random

# Template workflows (simplified for training)
TEMPLATES = {
    "basic_txt2img": {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["MODEL_PLACEHOLDER"]},
            {"id": 2, "type": "CLIPTextEncode", "widgets_values": ["PROMPT_PLACEHOLDER"]},
            {"id": 3, "type": "CLIPTextEncode", "widgets_values": ["NEGATIVE_PLACEHOLDER"]},
            {"id": 4, "type": "KSampler", "widgets_values": [0, "fixed", "STEPS_PLACEHOLDER", "CFG_PLACEHOLDER", "SAMPLER_PLACEHOLDER", "SCHEDULER_PLACEHOLDER", 1]},
            {"id": 5, "type": "EmptyLatentImage", "widgets_values": ["WIDTH_PLACEHOLDER", "HEIGHT_PLACEHOLDER", 1]},
            {"id": 6, "type": "VAEDecode"},
            {"id": 7, "type": "SaveImage"}
        ],
        "links": [
            [1, 1, 0, 4, 0, "MODEL"],
            [2, 1, 1, 2, 0, "CLIP"],
            [3, 1, 1, 3, 0, "CLIP"],
            [4, 1, 2, 6, 1, "VAE"],
            [5, 2, 0, 4, 1, "CONDITIONING"],
            [6, 3, 0, 4, 2, "CONDITIONING"],
            [7, 5, 0, 4, 3, "LATENT"],
            [8, 4, 0, 6, 0, "LATENT"],
            [9, 6, 0, 7, 0, "IMAGE"]
        ]
    },
    
    "with_lora": {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["MODEL_PLACEHOLDER"]},
            {"id": 2, "type": "LoraLoader", "widgets_values": ["LORA_PLACEHOLDER", "LORA_STRENGTH_PLACEHOLDER", "LORA_STRENGTH_PLACEHOLDER"]},
            {"id": 3, "type": "CLIPTextEncode", "widgets_values": ["PROMPT_PLACEHOLDER"]},
            {"id": 4, "type": "CLIPTextEncode", "widgets_values": ["NEGATIVE_PLACEHOLDER"]},
            {"id": 5, "type": "KSampler", "widgets_values": [0, "fixed", "STEPS_PLACEHOLDER", "CFG_PLACEHOLDER", "SAMPLER_PLACEHOLDER", "SCHEDULER_PLACEHOLDER", 1]},
            {"id": 6, "type": "EmptyLatentImage", "widgets_values": ["WIDTH_PLACEHOLDER", "HEIGHT_PLACEHOLDER", 1]},
            {"id": 7, "type": "VAEDecode"},
            {"id": 8, "type": "SaveImage"}
        ],
        "links": [
            [1, 1, 0, 2, 0, "MODEL"],
            [2, 1, 1, 2, 1, "CLIP"],
            [3, 2, 0, 5, 0, "MODEL"],
            [4, 2, 1, 3, 0, "CLIP"],
            [5, 2, 1, 4, 0, "CLIP"],
            [6, 3, 0, 5, 1, "CONDITIONING"],
            [7, 4, 0, 5, 2, "CONDITIONING"],
            [8, 6, 0, 5, 3, "LATENT"],
            [9, 5, 0, 7, 0, "LATENT"],
            [10, 1, 2, 7, 1, "VAE"],
            [11, 7, 0, 8, 0, "IMAGE"]
        ]
    },
    
    "txt2video": {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["VIDEO_MODEL_PLACEHOLDER"]},
            {"id": 2, "type": "CLIPTextEncode", "widgets_values": ["PROMPT_PLACEHOLDER"]},
            {"id": 3, "type": "CLIPTextEncode", "widgets_values": ["NEGATIVE_PLACEHOLDER"]},
            {"id": 4, "type": "VideoLinearCFGGuidance", "widgets_values": ["CFG_PLACEHOLDER"]},
            {"id": 5, "type": "EmptyLatentVideo", "widgets_values": ["WIDTH_PLACEHOLDER", "HEIGHT_PLACEHOLDER", "FRAMES_PLACEHOLDER", 1]},
            {"id": 6, "type": "KSampler", "widgets_values": [0, "fixed", "STEPS_PLACEHOLDER", "CFG_PLACEHOLDER", "SAMPLER_PLACEHOLDER", "SCHEDULER_PLACEHOLDER", 1]},
            {"id": 7, "type": "VAEDecode"},
            {"id": 8, "type": "VHS_VideoCombine", "widgets_values": ["FPS_PLACEHOLDER", 0, "output_video"]}
        ],
        "links": [
            [1, 1, 0, 6, 0, "MODEL"],
            [2, 1, 1, 2, 0, "CLIP"],
            [3, 1, 1, 3, 0, "CLIP"],
            [4, 2, 0, 4, 0, "CONDITIONING"],
            [5, 3, 0, 4, 1, "CONDITIONING"],
            [6, 4, 0, 6, 1, "CONDITIONING"],
            [7, 4, 1, 6, 2, "CONDITIONING"],
            [8, 5, 0, 6, 3, "LATENT"],
            [9, 6, 0, 7, 0, "LATENT"],
            [10, 1, 2, 7, 1, "VAE"],
            [11, 7, 0, 8, 0, "IMAGE"]
        ]
    },
    
    "txt2audio": {
        "nodes": [
            {"id": 1, "type": "AudioLDM2LoadModel", "widgets_values": ["AUDIO_MODEL_PLACEHOLDER"]},
            {"id": 2, "type": "AudioLDM2Sampler", "widgets_values": ["PROMPT_PLACEHOLDER", "NEGATIVE_PLACEHOLDER", "DURATION_PLACEHOLDER", "STEPS_PLACEHOLDER", "CFG_PLACEHOLDER", 0]},
            {"id": 3, "type": "SaveAudio", "widgets_values": ["output_audio"]}
        ],
        "links": [
            [1, 1, 0, 2, 0, "MODEL"],
            [2, 2, 0, 3, 0, "AUDIO"]
        ]
    },
    
    "img2img": {
        "nodes": [
            {"id": 1, "type": "LoadImage", "widgets_values": ["INPUT_IMAGE_PLACEHOLDER"]},
            {"id": 2, "type": "CheckpointLoaderSimple", "widgets_values": ["MODEL_PLACEHOLDER"]},
            {"id": 3, "type": "CLIPTextEncode", "widgets_values": ["PROMPT_PLACEHOLDER"]},
            {"id": 4, "type": "CLIPTextEncode", "widgets_values": ["NEGATIVE_PLACEHOLDER"]},
            {"id": 5, "type": "VAEEncode"},
            {"id": 6, "type": "KSampler", "widgets_values": [0, "fixed", "STEPS_PLACEHOLDER", "CFG_PLACEHOLDER", "SAMPLER_PLACEHOLDER", "SCHEDULER_PLACEHOLDER", "DENOISE_PLACEHOLDER"]},
            {"id": 7, "type": "VAEDecode"},
            {"id": 8, "type": "SaveImage"}
        ],
        "links": [
            [1, 1, 0, 5, 0, "IMAGE"],
            [2, 2, 0, 6, 0, "MODEL"],
            [3, 2, 1, 3, 0, "CLIP"],
            [4, 2, 1, 4, 0, "CLIP"],
            [5, 2, 2, 5, 1, "VAE"],
            [6, 2, 2, 7, 1, "VAE"],
            [7, 3, 0, 6, 1, "CONDITIONING"],
            [8, 4, 0, 6, 2, "CONDITIONING"],
            [9, 5, 0, 6, 3, "LATENT"],
            [10, 6, 0, 7, 0, "LATENT"],
            [11, 7, 0, 8, 0, "IMAGE"]
        ]
    },
    
    "img2video": {
        "nodes": [
            {"id": 1, "type": "LoadImage", "widgets_values": ["INPUT_IMAGE_PLACEHOLDER"]},
            {"id": 2, "type": "CheckpointLoaderSimple", "widgets_values": ["VIDEO_MODEL_PLACEHOLDER"]},
            {"id": 3, "type": "CLIPTextEncode", "widgets_values": ["PROMPT_PLACEHOLDER"]},
            {"id": 4, "type": "CLIPTextEncode", "widgets_values": ["NEGATIVE_PLACEHOLDER"]},
            {"id": 5, "type": "VAEEncode"},
            {"id": 6, "type": "VideoLinearCFGGuidance", "widgets_values": ["CFG_PLACEHOLDER"]},
            {"id": 7, "type": "SVD_img2vid_Conditioning", "widgets_values": ["WIDTH_PLACEHOLDER", "HEIGHT_PLACEHOLDER", "FRAMES_PLACEHOLDER", "MOTION_PLACEHOLDER", "AUGMENTATION_PLACEHOLDER"]},
            {"id": 8, "type": "KSampler", "widgets_values": [0, "fixed", "STEPS_PLACEHOLDER", "CFG_PLACEHOLDER", "SAMPLER_PLACEHOLDER", "SCHEDULER_PLACEHOLDER", 1]},
            {"id": 9, "type": "VAEDecode"},
            {"id": 10, "type": "VHS_VideoCombine", "widgets_values": ["FPS_PLACEHOLDER", 0, "output_video"]}
        ],
        "links": [
            [1, 1, 0, 5, 0, "IMAGE"],
            [2, 1, 0, 7, 0, "IMAGE"],
            [3, 2, 0, 8, 0, "MODEL"],
            [4, 2, 1, 3, 0, "CLIP"],
            [5, 2, 1, 4, 0, "CLIP"],
            [6, 2, 2, 5, 1, "VAE"],
            [7, 2, 2, 9, 1, "VAE"],
            [8, 3, 0, 6, 0, "CONDITIONING"],
            [9, 4, 0, 6, 1, "CONDITIONING"],
            [10, 6, 0, 8, 1, "CONDITIONING"],
            [11, 6, 1, 8, 2, "CONDITIONING"],
            [12, 7, 0, 8, 3, "LATENT"],
            [13, 8, 0, 9, 0, "LATENT"],
            [14, 9, 0, 10, 0, "IMAGE"]
        ]
    },
    
    "video2video": {
        "nodes": [
            {"id": 1, "type": "VHS_LoadVideo", "widgets_values": ["INPUT_VIDEO_PLACEHOLDER"]},
            {"id": 2, "type": "CheckpointLoaderSimple", "widgets_values": ["VIDEO_MODEL_PLACEHOLDER"]},
            {"id": 3, "type": "CLIPTextEncode", "widgets_values": ["PROMPT_PLACEHOLDER"]},
            {"id": 4, "type": "CLIPTextEncode", "widgets_values": ["NEGATIVE_PLACEHOLDER"]},
            {"id": 5, "type": "VAEEncodeBatched"},
            {"id": 6, "type": "VideoLinearCFGGuidance", "widgets_values": ["CFG_PLACEHOLDER"]},
            {"id": 7, "type": "KSampler", "widgets_values": [0, "fixed", "STEPS_PLACEHOLDER", "CFG_PLACEHOLDER", "SAMPLER_PLACEHOLDER", "SCHEDULER_PLACEHOLDER", "DENOISE_PLACEHOLDER"]},
            {"id": 8, "type": "VAEDecode"},
            {"id": 9, "type": "VHS_VideoCombine", "widgets_values": ["FPS_PLACEHOLDER", 0, "output_video"]}
        ],
        "links": [
            [1, 1, 0, 5, 0, "IMAGE"],
            [2, 2, 0, 7, 0, "MODEL"],
            [3, 2, 1, 3, 0, "CLIP"],
            [4, 2, 1, 4, 0, "CLIP"],
            [5, 2, 2, 5, 1, "VAE"],
            [6, 2, 2, 8, 1, "VAE"],
            [7, 3, 0, 6, 0, "CONDITIONING"],
            [8, 4, 0, 6, 1, "CONDITIONING"],
            [9, 6, 0, 7, 1, "CONDITIONING"],
            [10, 6, 1, 7, 2, "CONDITIONING"],
            [11, 5, 0, 7, 3, "LATENT"],
            [12, 7, 0, 8, 0, "LATENT"],
            [13, 8, 0, 9, 0, "IMAGE"]
        ]
    }
}

# Variation options
MODELS = ["sd-v1-5.safetensors", "sd-v2-1.safetensors", "sdxl-base.safetensors"]
VIDEO_MODELS = ["svd-v1.safetensors", "svd-xt.safetensors", "animatediff-v3.safetensors"]
AUDIO_MODELS = ["audioldm2-large.safetensors", "audioldm2-music.safetensors"]
LORAS = ["detail-tweaker.safetensors", "add-detail.safetensors", "style-enhancer.safetensors"]
SAMPLERS = ["euler", "euler_a", "dpm_2", "dpm_2_ancestral", "dpmpp_2m", "dpmpp_sde"]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform"]
SIZES = [(512, 512), (768, 768), (1024, 1024), (512, 768), (768, 512)]
VIDEO_SIZES = [(512, 512), (768, 512), (1024, 576)]
STEPS_RANGE = [15, 20, 25, 30, 40, 50]
CFG_RANGE = [5, 6, 7, 8, 9, 10, 11]
FRAMES_RANGE = [16, 24, 32, 48, 64]
FPS_RANGE = [8, 12, 16, 24, 30]
DENOISE_RANGE = [0.5, 0.6, 0.7, 0.8, 0.9]
MOTION_RANGE = [64, 127, 191, 255]
AUGMENTATION_RANGE = [0.0, 0.1, 0.2]
AUDIO_DURATION_RANGE = [5, 10, 15, 20, 30]

# Example prompts (you'll want to expand this significantly)
PROMPTS = [
    "a beautiful sunset over mountains",
    "a cat sitting on a windowsill",
    "a futuristic cityscape at night",
    "a portrait of an elderly person",
    "a fantasy dragon in flight",
    "a serene lake with reflections",
    "a bustling marketplace",
    "an astronaut floating in space",
    "a cozy cabin in the woods",
    "a vibrant flower garden"
]

VIDEO_PROMPTS = [
    "waves crashing on a beach",
    "clouds moving across the sky",
    "a person walking through a forest",
    "traffic flowing through city streets",
    "a bird flying through the air",
    "rain falling on a window",
    "fire burning in a fireplace",
    "flowers blooming in timelapse"
]

AUDIO_PROMPTS = [
    "gentle rain falling on leaves",
    "ocean waves on a beach",
    "birds chirping in a forest",
    "soft piano melody",
    "ambient electronic music",
    "thunder and lightning storm",
    "crackling fireplace",
    "wind blowing through trees"
]

IMG2IMG_PROMPTS = [
    "transform into oil painting style",
    "add dramatic lighting",
    "make it look like a sketch",
    "enhance colors and details",
    "convert to anime style",
    "add fantasy elements"
]

NEGATIVE_PROMPTS = [
    "blurry, low quality",
    "distorted, ugly",
    "bad anatomy, deformed",
    "watermark, text",
    "oversaturated, unrealistic"
]

INPUT_IMAGES = ["input_001.png", "input_002.png", "input_003.png", "sample.jpg"]
INPUT_VIDEOS = ["input_video_001.mp4", "input_video_002.mp4", "sample_video.mp4"]

def create_basic_txt2img_example():
    """Create a basic text-to-image example"""
    
    # Random selections
    model = random.choice(MODELS)
    prompt = random.choice(PROMPTS)
    negative = random.choice(NEGATIVE_PROMPTS)
    sampler = random.choice(SAMPLERS)
    scheduler = random.choice(SCHEDULERS)
    steps = random.choice(STEPS_RANGE)
    cfg = random.choice(CFG_RANGE)
    width, height = random.choice(SIZES)
    
    # Create workflow from template
    workflow = json.loads(json.dumps(TEMPLATES["basic_txt2img"]))
    
    # Replace placeholders
    workflow["nodes"][0]["widgets_values"][0] = model
    workflow["nodes"][1]["widgets_values"][0] = prompt
    workflow["nodes"][2]["widgets_values"][0] = negative
    workflow["nodes"][3]["widgets_values"][2] = steps
    workflow["nodes"][3]["widgets_values"][3] = cfg
    workflow["nodes"][3]["widgets_values"][4] = sampler
    workflow["nodes"][3]["widgets_values"][5] = scheduler
    workflow["nodes"][4]["widgets_values"][0] = width
    workflow["nodes"][4]["widgets_values"][1] = height
    
    # Create training example
    example = {
        "instruction": "Generate a ComfyUI workflow for text-to-image generation",
        "input": f"Model: {model.replace('.safetensors', '')}, Prompt: '{prompt}', Negative: '{negative}', Sampler: {sampler}, Scheduler: {scheduler}, Steps: {steps}, CFG: {cfg}, Size: {width}x{height}",
        "output": json.dumps(workflow, separators=(',', ':'))
    }
    
    return example

def create_lora_example():
    """Create a LoRA workflow example"""
    
    # Random selections
    model = random.choice(MODELS)
    lora = random.choice(LORAS)
    lora_strength = round(random.uniform(0.5, 1.0), 1)
    prompt = random.choice(PROMPTS)
    negative = random.choice(NEGATIVE_PROMPTS)
    sampler = random.choice(SAMPLERS)
    scheduler = random.choice(SCHEDULERS)
    steps = random.choice(STEPS_RANGE)
    cfg = random.choice(CFG_RANGE)
    width, height = random.choice(SIZES)
    
    # Create workflow from template
    workflow = json.loads(json.dumps(TEMPLATES["with_lora"]))
    
    # Replace placeholders
    workflow["nodes"][0]["widgets_values"][0] = model
    workflow["nodes"][1]["widgets_values"][0] = lora
    workflow["nodes"][1]["widgets_values"][1] = lora_strength
    workflow["nodes"][1]["widgets_values"][2] = lora_strength
    workflow["nodes"][2]["widgets_values"][0] = prompt
    workflow["nodes"][3]["widgets_values"][0] = negative
    workflow["nodes"][4]["widgets_values"][2] = steps
    workflow["nodes"][4]["widgets_values"][3] = cfg
    workflow["nodes"][4]["widgets_values"][4] = sampler
    workflow["nodes"][4]["widgets_values"][5] = scheduler
    workflow["nodes"][5]["widgets_values"][0] = width
    workflow["nodes"][5]["widgets_values"][1] = height
    
    # Create training example
    example = {
        "instruction": "Create a ComfyUI workflow with LoRA",
        "input": f"Base model: {model.replace('.safetensors', '')}, LoRA: {lora.replace('.safetensors', '')}, LoRA strength: {lora_strength}, Prompt: '{prompt}', Negative: '{negative}', Sampler: {sampler}, Steps: {steps}, CFG: {cfg}, Size: {width}x{height}",
        "output": json.dumps(workflow, separators=(',', ':'))
    }
    
    return example

def create_txt2video_example():
    """Create a text-to-video workflow example"""
    
    # Random selections
    model = random.choice(VIDEO_MODELS)
    prompt = random.choice(VIDEO_PROMPTS)
    negative = random.choice(NEGATIVE_PROMPTS)
    sampler = random.choice(SAMPLERS)
    scheduler = random.choice(SCHEDULERS)
    steps = random.choice(STEPS_RANGE)
    cfg = random.choice(CFG_RANGE)
    width, height = random.choice(VIDEO_SIZES)
    frames = random.choice(FRAMES_RANGE)
    fps = random.choice(FPS_RANGE)
    
    # Create workflow from template
    workflow = json.loads(json.dumps(TEMPLATES["txt2video"]))
    
    # Replace placeholders
    workflow["nodes"][0]["widgets_values"][0] = model
    workflow["nodes"][1]["widgets_values"][0] = prompt
    workflow["nodes"][2]["widgets_values"][0] = negative
    workflow["nodes"][3]["widgets_values"][0] = cfg
    workflow["nodes"][4]["widgets_values"][0] = width
    workflow["nodes"][4]["widgets_values"][1] = height
    workflow["nodes"][4]["widgets_values"][2] = frames
    workflow["nodes"][5]["widgets_values"][2] = steps
    workflow["nodes"][5]["widgets_values"][3] = cfg
    workflow["nodes"][5]["widgets_values"][4] = sampler
    workflow["nodes"][5]["widgets_values"][5] = scheduler
    workflow["nodes"][7]["widgets_values"][0] = fps
    
    # Create training example
    example = {
        "instruction": "Generate a ComfyUI workflow for text-to-video generation",
        "input": f"Model: {model.replace('.safetensors', '')}, Prompt: '{prompt}', Negative: '{negative}', Sampler: {sampler}, Scheduler: {scheduler}, Steps: {steps}, CFG: {cfg}, Size: {width}x{height}, Frames: {frames}, FPS: {fps}",
        "output": json.dumps(workflow, separators=(',', ':'))
    }
    
    return example

def create_txt2audio_example():
    """Create a text-to-audio workflow example"""
    
    # Random selections
    model = random.choice(AUDIO_MODELS)
    prompt = random.choice(AUDIO_PROMPTS)
    negative = random.choice(NEGATIVE_PROMPTS)
    duration = random.choice(AUDIO_DURATION_RANGE)
    steps = random.choice(STEPS_RANGE)
    cfg = random.choice(CFG_RANGE)
    
    # Create workflow from template
    workflow = json.loads(json.dumps(TEMPLATES["txt2audio"]))
    
    # Replace placeholders
    workflow["nodes"][0]["widgets_values"][0] = model
    workflow["nodes"][1]["widgets_values"][0] = prompt
    workflow["nodes"][1]["widgets_values"][1] = negative
    workflow["nodes"][1]["widgets_values"][2] = duration
    workflow["nodes"][1]["widgets_values"][3] = steps
    workflow["nodes"][1]["widgets_values"][4] = cfg
    
    # Create training example
    example = {
        "instruction": "Create a ComfyUI workflow for text-to-audio generation",
        "input": f"Model: {model.replace('.safetensors', '')}, Prompt: '{prompt}', Negative: '{negative}', Duration: {duration}s, Steps: {steps}, CFG: {cfg}",
        "output": json.dumps(workflow, separators=(',', ':'))
    }
    
    return example

def create_img2img_example():
    """Create an image-to-image workflow example"""
    
    # Random selections
    input_image = random.choice(INPUT_IMAGES)
    model = random.choice(MODELS)
    prompt = random.choice(IMG2IMG_PROMPTS)
    negative = random.choice(NEGATIVE_PROMPTS)
    sampler = random.choice(SAMPLERS)
    scheduler = random.choice(SCHEDULERS)
    steps = random.choice(STEPS_RANGE)
    cfg = random.choice(CFG_RANGE)
    denoise = random.choice(DENOISE_RANGE)
    
    # Create workflow from template
    workflow = json.loads(json.dumps(TEMPLATES["img2img"]))
    
    # Replace placeholders
    workflow["nodes"][0]["widgets_values"][0] = input_image
    workflow["nodes"][1]["widgets_values"][0] = model
    workflow["nodes"][2]["widgets_values"][0] = prompt
    workflow["nodes"][3]["widgets_values"][0] = negative
    workflow["nodes"][5]["widgets_values"][2] = steps
    workflow["nodes"][5]["widgets_values"][3] = cfg
    workflow["nodes"][5]["widgets_values"][4] = sampler
    workflow["nodes"][5]["widgets_values"][5] = scheduler
    workflow["nodes"][5]["widgets_values"][6] = denoise
    
    # Create training example
    example = {
        "instruction": "Generate a ComfyUI workflow for image-to-image transformation",
        "input": f"Input image: {input_image}, Model: {model.replace('.safetensors', '')}, Prompt: '{prompt}', Negative: '{negative}', Sampler: {sampler}, Scheduler: {scheduler}, Steps: {steps}, CFG: {cfg}, Denoise: {denoise}",
        "output": json.dumps(workflow, separators=(',', ':'))
    }
    
    return example

def create_img2video_example():
    """Create an image-to-video workflow example"""
    
    # Random selections
    input_image = random.choice(INPUT_IMAGES)
    model = random.choice(VIDEO_MODELS)
    prompt = random.choice(VIDEO_PROMPTS)
    negative = random.choice(NEGATIVE_PROMPTS)
    sampler = random.choice(SAMPLERS)
    scheduler = random.choice(SCHEDULERS)
    steps = random.choice(STEPS_RANGE)
    cfg = random.choice(CFG_RANGE)
    width, height = random.choice(VIDEO_SIZES)
    frames = random.choice(FRAMES_RANGE)
    fps = random.choice(FPS_RANGE)
    motion = random.choice(MOTION_RANGE)
    augmentation = random.choice(AUGMENTATION_RANGE)
    
    # Create workflow from template
    workflow = json.loads(json.dumps(TEMPLATES["img2video"]))
    
    # Replace placeholders
    workflow["nodes"][0]["widgets_values"][0] = input_image
    workflow["nodes"][1]["widgets_values"][0] = model
    workflow["nodes"][2]["widgets_values"][0] = prompt
    workflow["nodes"][3]["widgets_values"][0] = negative
    workflow["nodes"][5]["widgets_values"][0] = cfg
    workflow["nodes"][6]["widgets_values"][0] = width
    workflow["nodes"][6]["widgets_values"][1] = height
    workflow["nodes"][6]["widgets_values"][2] = frames
    workflow["nodes"][6]["widgets_values"][3] = motion
    workflow["nodes"][6]["widgets_values"][4] = augmentation
    workflow["nodes"][7]["widgets_values"][2] = steps
    workflow["nodes"][7]["widgets_values"][3] = cfg
    workflow["nodes"][7]["widgets_values"][4] = sampler
    workflow["nodes"][7]["widgets_values"][5] = scheduler
    workflow["nodes"][9]["widgets_values"][0] = fps
    
    # Create training example
    example = {
        "instruction": "Create a ComfyUI workflow for image-to-video generation",
        "input": f"Input image: {input_image}, Model: {model.replace('.safetensors', '')}, Prompt: '{prompt}', Negative: '{negative}', Sampler: {sampler}, Scheduler: {scheduler}, Steps: {steps}, CFG: {cfg}, Size: {width}x{height}, Frames: {frames}, FPS: {fps}, Motion: {motion}, Augmentation: {augmentation}",
        "output": json.dumps(workflow, separators=(',', ':'))
    }
    
    return example

def create_video2video_example():
    """Create a video-to-video workflow example"""
    
    # Random selections
    input_video = random.choice(INPUT_VIDEOS)
    model = random.choice(VIDEO_MODELS)
    prompt = random.choice(VIDEO_PROMPTS)
    negative = random.choice(NEGATIVE_PROMPTS)
    sampler = random.choice(SAMPLERS)
    scheduler = random.choice(SCHEDULERS)
    steps = random.choice(STEPS_RANGE)
    cfg = random.choice(CFG_RANGE)
    fps = random.choice(FPS_RANGE)
    denoise = random.choice(DENOISE_RANGE)
    
    # Create workflow from template
    workflow = json.loads(json.dumps(TEMPLATES["video2video"]))
    
    # Replace placeholders
    workflow["nodes"][0]["widgets_values"][0] = input_video
    workflow["nodes"][1]["widgets_values"][0] = model
    workflow["nodes"][2]["widgets_values"][0] = prompt
    workflow["nodes"][3]["widgets_values"][0] = negative
    workflow["nodes"][5]["widgets_values"][0] = cfg
    workflow["nodes"][6]["widgets_values"][2] = steps
    workflow["nodes"][6]["widgets_values"][3] = cfg
    workflow["nodes"][6]["widgets_values"][4] = sampler
    workflow["nodes"][6]["widgets_values"][5] = scheduler
    workflow["nodes"][6]["widgets_values"][6] = denoise
    workflow["nodes"][8]["widgets_values"][0] = fps
    
    # Create training example
    example = {
        "instruction": "Generate a ComfyUI workflow for video-to-video transformation",
        "input": f"Input video: {input_video}, Model: {model.replace('.safetensors', '')}, Prompt: '{prompt}', Negative: '{negative}', Sampler: {sampler}, Scheduler: {scheduler}, Steps: {steps}, CFG: {cfg}, FPS: {fps}, Denoise: {denoise}",
        "output": json.dumps(workflow, separators=(',', ':'))
    }
    
    return example

def generate_dataset(num_examples=500):
    """Generate a dataset with mixed workflow types"""
    
    dataset = []
    
    # Distribution of workflow types
    # 30% text-to-image (basic + LoRA)
    # 20% text-to-video
    # 10% text-to-audio
    # 15% image-to-image
    # 15% image-to-video
    # 10% video-to-video
    
    num_txt2img_basic = int(num_examples * 0.20)
    num_txt2img_lora = int(num_examples * 0.10)
    num_txt2video = int(num_examples * 0.20)
    num_txt2audio = int(num_examples * 0.10)
    num_img2img = int(num_examples * 0.15)
    num_img2video = int(num_examples * 0.15)
    num_video2video = num_examples - (num_txt2img_basic + num_txt2img_lora + num_txt2video + num_txt2audio + num_img2img + num_img2video)
    
    print(f"Generating {num_examples} examples...")
    print(f"  - {num_txt2img_basic} text-to-image (basic)")
    print(f"  - {num_txt2img_lora} text-to-image (with LoRA)")
    print(f"  - {num_txt2video} text-to-video")
    print(f"  - {num_txt2audio} text-to-audio")
    print(f"  - {num_img2img} image-to-image")
    print(f"  - {num_img2video} image-to-video")
    print(f"  - {num_video2video} video-to-video")
    print()
    
    # Generate text-to-image (basic)
    print("Generating text-to-image (basic)...")
    for i in range(num_txt2img_basic):
        dataset.append(create_basic_txt2img_example())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_txt2img_basic}")
    
    # Generate text-to-image (with LoRA)
    print("Generating text-to-image (with LoRA)...")
    for i in range(num_txt2img_lora):
        dataset.append(create_lora_example())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_txt2img_lora}")
    
    # Generate text-to-video
    print("Generating text-to-video...")
    for i in range(num_txt2video):
        dataset.append(create_txt2video_example())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_txt2video}")
    
    # Generate text-to-audio
    print("Generating text-to-audio...")
    for i in range(num_txt2audio):
        dataset.append(create_txt2audio_example())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_txt2audio}")
    
    # Generate image-to-image
    print("Generating image-to-image...")
    for i in range(num_img2img):
        dataset.append(create_img2img_example())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_img2img}")
    
    # Generate image-to-video
    print("Generating image-to-video...")
    for i in range(num_img2video):
        dataset.append(create_img2video_example())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_img2video}")
    
    # Generate video-to-video
    print("Generating video-to-video...")
    for i in range(num_video2video):
        dataset.append(create_video2video_example())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_video2video}")
    
    # Shuffle
    random.shuffle(dataset)
    
    return dataset

def save_dataset(dataset, filename="comfyui_dataset.jsonl"):
    """Save dataset to JSONL file"""
    
    with open(filename, "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")
    
    print(f"\n✓ Saved {len(dataset)} examples to {filename}")

def preview_examples(dataset, num=3):
    """Preview some examples"""
    
    print("\n" + "="*70)
    print("DATASET PREVIEW")
    print("="*70)
    
    for i, example in enumerate(dataset[:num]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Input: {example['input'][:100]}...")
        print(f"Output length: {len(example['output'])} characters")
        print("-"*70)

if __name__ == "__main__":
    print("="*70)
    print("COMFYUI DATASET GENERATOR - MULTI-MODAL WORKFLOWS")
    print("="*70)
    print("\nThis script generates training examples for ComfyUI workflow generation.")
    print("It creates variations of workflows for:")
    print("  • Text-to-Image (basic and with LoRA)")
    print("  • Text-to-Video")
    print("  • Text-to-Audio")
    print("  • Image-to-Image")
    print("  • Image-to-Video")
    print("  • Video-to-Video")
    print("\nNote: This is a STARTER dataset. You should:")
    print("  1. Review and customize the prompts")
    print("  2. Adjust workflow templates to match your ComfyUI setup")
    print("  3. Ensure examples match your use case")
    print("="*70)
    
    # Generate dataset
    num_examples = 500  # Increased to 500 for better coverage
    dataset = generate_dataset(num_examples)
    
    # Preview
    preview_examples(dataset, num=5)
    
    # Save
    save_dataset(dataset, "comfyui_dataset.jsonl")
    
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    # Count workflow types
    workflow_types = {}
    for example in dataset:
        instruction = example["instruction"]
        workflow_types[instruction] = workflow_types.get(instruction, 0) + 1
    
    for workflow_type, count in sorted(workflow_types.items()):
        percentage = (count / len(dataset)) * 100
        print(f"{workflow_type}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review comfyui_dataset.jsonl")
    print("2. Customize prompts and add more variations")
    print("3. Test a few workflows in ComfyUI to verify correctness")
    print("4. Run validate_comfyui_dataset.py to check for issues")
    print("5. Proceed to Phase 5: Dataset Formatting")
    print("="*70)
