

"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""
import argparse
import logging
from typing import Literal, Optional
import torch
import json
from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

import sys
import os
# Get the absolute path of the current file (e.g., .../project/inference)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (the parent directory of 'inference')
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Add the project root directory to Python's module search path
# This ensures imports such as 'from finetune.datasets.utils import ...' work,
# regardless of where the script is executed.
sys.path.append(ROOT_DIR)


from finetune.datasets.utils import preprocess_video_with_resize
from finetune.utils import (
    free_memory,
)
from torchvision import transforms
from pathlib import Path
import hashlib
from PIL import Image
import os
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
import cv2
os.environ['TOKENIZERS_PARALLELISM']="1"

logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}

pipe = None

def encode_video(vae=None, video: torch.Tensor = None) -> torch.Tensor: 
    # shape of input video: [B, C, F, H, W]
    video = video.to(vae.device, dtype=vae.dtype) 
    latent_dist = vae.encode(video).latent_dist 
    latent = latent_dist.sample() * vae.config.scaling_factor 
    free_memory()
    return latent

def preprocess(video_path: Path,max_num_frames,height,width,key='',target_fps=-1) -> torch.Tensor:
    return preprocess_video_with_resize(
        video_path,
        max_num_frames,
        height,
        width,
        key=key,
        target_fps=target_fps
    )

def embed_video(vae,video_cond_path,max_num_frames,height,width,key='',target_fps=-1) -> torch.Tensor: 
    frames = preprocess(video_cond_path,max_num_frames,height,width,key=key,target_fps=target_fps)
    save_frames = frames
    frames = frames.to("cuda") 
    # Current shape of frames: [F, C, H, W]
    frames = video_transform(frames) 
    # Convert to [B, C, F, H, W]
    frames = frames.unsqueeze(0)
    frames = frames.permute(0, 2, 1, 3, 4).contiguous() 
    encoded_video = encode_video(vae,frames) 

    # [1, C, F, H, W] -> [C, F, H, W]
    encoded_video = encoded_video[0]

    return encoded_video,save_frames

def video_transform(frames: torch.Tensor) -> torch.Tensor:
    __frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
    return torch.stack([__frame_transform(f) for f in frames], dim=0)

def generate_video(
    meta_path:list,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    mmc_args: Optional[Dict[str, Any]] = None,
    args = None,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.
    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """
    
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    global pipe 

    image = None
    video = None

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    pipe.to("cuda") 

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload() # slow inference
    
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.vae.requires_grad_(False)

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]

    elif generate_type == "t2v":

        task_keys = args.task_keys
        from tqdm import tqdm

        cal_keys = ['canny','blur','lr']
        for meta in tqdm(meta_path,desc='infer...'):
  
                prompt = meta['prompt']
                prompt_filename = prompt[:25]
                # Calculate hash of reversed prompt as a unique identifier
                reversed_prompt = prompt[::-1]
                extension = "mp4"
                hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]
                file_name = f"validation-{prompt_filename}-{hash_suffix}_merge_{mmc_args['idx_cond_modality']}_seed={seed}.{extension}"
                temp_name = file_name
                file_name = os.path.join(args.output_dir, file_name)
                eval_dict = {}
                
                if os.path.exists(file_name):
                    continue

                os.makedirs(args.output_dir, exist_ok=True)

                if mmc_args['idx_cond_modality'] >=0:
                    key = task_keys[mmc_args['idx_cond_modality']]
                    cond_path = meta['modal'].get(key,None)
                    cal_key = '' 
                    if cond_path is None and key in cal_keys:
                        cond_path = meta['modal']['rgb']
                        cal_key = key
                    encoded_video,save_frames = embed_video(pipe.vae,cond_path,num_frames,height,width,key=cal_key,target_fps=fps)
                    mmc_args['latent_cond_modal'] = encoded_video
                    save_frames = save_frames.permute(0,2,3,1) # image: [T,C,H,W]
                    input_video = np.array(save_frames).astype(np.uint8) # uint8


                video_generate = pipe(
                    height=height,
                    width=width,
                    prompt=prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    num_inference_steps=num_inference_steps, 
                    num_frames=num_frames,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
                    task_keys=task_keys,
                    mmc_args=mmc_args
                )

                validation_artifacts_list = video_generate  # validation result
            

                combined_array = []
                labels = []

                for validation_dict in validation_artifacts_list:

                    key, value = list(validation_dict.items())[0]
                    combined_array.append(np.array(value['frames'][0]))
                    labels.append(key)
            

                if mmc_args['idx_cond_modality'] >=0: # use condition
                    combined_array.append(input_video)
                    labels.append('condition')
                

                file_name = f"validation-{prompt_filename}-{hash_suffix}_merge_{mmc_args['idx_cond_modality']}_seed={seed}.{extension}"
                file_name = os.path.join(args.output_dir, file_name)     
                video_output = add_labels_to_video_grid(combined_array, labels) # merge video
                video_output = [Image.fromarray(frame) for frame in video_output]
                export_to_video(video_output, file_name, fps=fps)

                print("[INFO] save to ",file_name)

def save_metadata(path_meta_data, lis_meta_data):
    print(f'Save samples: {len(lis_meta_data)}')
    with open(path_meta_data, "w") as f:
        for meta in lis_meta_data:
            f.write(json.dumps(meta) + "\n")

def load_metadata(metadata_path,shuffle=False,samples=-1):
    metas = []
    cnt = 0
    print(f"[INFO] Processing {metadata_path}")
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                metas.append(json.loads(line))
                cnt += 1
        print(f"{metadata_path} {cnt} samples")
    except UnicodeDecodeError:
        print("loading exist error!")
    if samples==-1:
        metas = metas
    else:
        metas = metas[:samples] if samples < cnt else metas
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(metas)
    return metas

def add_labels_to_video(video_list, labels, font_scale=0.75, thickness=1):
    """
    :param video_list: List[np.ndarray],
    :param labels: List[str],
    :param font_scale,
    :param thicknes,
    :return: (T, H+label_height, W, C)
    """
    assert len(video_list) == len(labels), "video list == labels"

    print(f"video_list[0].shape: {video_list[0].shape}")
    T, H, W, C = video_list[0].shape

    T = min([video.shape[0] for video in video_list])
    video_list = [video[:T] for video in video_list]

    video_concat = np.concatenate(video_list, axis=2)  

    label_height = 32
    label_area = np.ones((T, label_height, video_concat.shape[2], 3), dtype=np.uint8)

    section_width = W 
    for i, label in enumerate(labels):
        x_pos = i * section_width + section_width // 2 - 16  
        y_pos = label_height - 10  
        cv2.putText(label_area[0], label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    for i in range(T):
        label_area[i] = label_area[0]

    video_with_labels = np.concatenate([label_area,video_concat], axis=1)

    return video_with_labels

def add_labels_to_video_grid(video_list, labels, font_scale=0.75, thickness=1, max_per_row=3):
    """
    :param video_list: List[np.ndarray],(T, H, W, 3)
    :param labels: List[str],
    :param font_scale: 
    :param thickness: 
    :param max_per_row: 
    :return: (T, H_total, W_total, 3)
    """
    assert len(video_list) == len(labels), "video list == labels"
    T, H, W, C = video_list[0].shape
    label_height = 32

    num_videos = len(video_list)
    import math
    num_rows = math.ceil(num_videos / max_per_row)

    labeled_videos = []

    for idx, (video, label) in enumerate(zip(video_list, labels)):
        
        label_area = np.zeros((T, label_height, W, 3), dtype=np.uint8)

        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x_pos = (W - text_width) // 2
        y_pos = (label_height + text_height) // 2 - 2  

        cv2.putText(label_area[0], label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        for t in range(1, T):
            label_area[t] = label_area[0]

        labeled_video = np.concatenate([label_area, video], axis=1)  # shape (T, H+label_height, W, 3)
        labeled_videos.append(labeled_video)

    row_videos = []
    for i in range(num_rows):
        row = labeled_videos[i * max_per_row : (i + 1) * max_per_row]
        if len(row) < max_per_row:
            pad_count = max_per_row - len(row)
            blank_video = np.zeros_like(labeled_videos[0])
            row.extend([blank_video] * pad_count)
        row_concat = np.concatenate(row, axis=2)
        row_videos.append(row_concat)

    final_video = np.concatenate(row_videos, axis=1)
    return final_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="Pipeline Path", help="Path of the pre-trained model use"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps") 
    parser.add_argument("--num_frames", type=int, default=49, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=640, help="The width of the generated video")
    parser.add_argument("--height", type=int, default=480, help="The height of the generated video")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=999999, help="The seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./debug_output", help="The path save generated video")


    ## omni args
    parser.add_argument("--task_keys", type=str, nargs="+", default=["rgb","depth","canny","segment"]) 
    parser.add_argument("--idx_cond_modality", type=int, default=0, help="The index of condition modality,-1 no condition, 0:rgb, 1:depth, 2:canny, 3:segment") 
    parser.add_argument("--use_modal_emb_condgen", action='store_true', help="Whether to use modal embeddings for condition generation") 
    parser.add_argument("--meta_path", type=str, default="../dataset/data.jsonl", help="The meta path")
    parser.add_argument("--samples", type=int, default=-1, help="The sample video number.")


    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    args.use_modal_emb_condgen = True

    mmc_args = {
        "use_modal_emb_condgen":args.use_modal_emb_condgen,
        "idx_cond_modality": args.idx_cond_modality, 
    }

    mode_maps = ['t2mm','rgb_cond','depth_cond','canny_cond','segment_cond']
    print("[INFO] idx_cond_modality:",mmc_args["idx_cond_modality"])
    print("[INFO] !!! curr mode:",mode_maps[mmc_args["idx_cond_modality"]+1])

    generate_video(
        meta_path = load_metadata(args.meta_path,samples=args.samples),
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
        mmc_args = mmc_args,
        args = args
    )
