import logging
from pathlib import Path
from typing import List, Tuple
import PIL.Image as Image
import cv2
import torch
import matplotlib.cm as cm
import numpy as np
from torchvision.transforms.functional import resize
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except:
    import cv2
    DECORD_AVAILABLE = False

assert DECORD_AVAILABLE == True

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch") 


##########  loaders  ##########


def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_images(image_path: Path) -> List[Path]:
    with open(image_path, "r", encoding="utf-8") as file:
        return [image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_images_from_videos(videos_path: List[Path]) -> List[Path]:
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        frame_path = first_frames_dir / f"{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with same name as video
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths


##########  preprocessors  ##########

def transfer_canny_image(image): # generate data

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, 75, 150) 

    edges_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return edges_image

def transfer_canny_flip_image(image): # generate data

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, 75, 150) 

    edges = 255 - edges

    edges_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return edges_image

def transfer_blur_image(image): # generate data
    kernel_size = (15, 15)
    blur_image = cv2.blur(image,kernel_size)
    return blur_image

def transfer_lr_image(image): # generate data


    original_height, original_width = image.shape[:2]

    scale = 0.1  
    low_res_width = int(original_width * scale)
    low_res_height = int(original_height * scale)

    low_res_image = cv2.resize(image, (low_res_width, low_res_height), interpolation=cv2.INTER_NEAREST)


    restored_image = cv2.resize(low_res_image, (original_width, original_height), interpolation=cv2.INTER_NEAREST)


    return restored_image

def transfer_depth_color_map(frames,type='inferno'): 
    import imageio
    colormap = np.array(cm.get_cmap(type).colors)
    d_min, d_max = frames.min(), frames.max()
    depths = []
    for i in range(frames.shape[0]):
        depth = frames[i]
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
        depths.append(depth_vis)
    depths = np.array(depths)
    depths = torch.from_numpy(depths)
    return depths

def preprocess_image_with_resize(
    image_path: Path | str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single image.

    Args:
        image_path: Path to the image file.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] where:
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).contiguous()
    return image


def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
    key:str = '',
    target_fps: int = -1,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """

    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)


    fps = video_reader.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(video_reader.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(video_reader), stride))
    if max_num_frames != -1 and max_num_frames < len(frames_idx):
        frames_idx = frames_idx[:max_num_frames]
    frames = video_reader.get_batch(frames_idx) 
    video_num_frames = len(frames)


    if key == 'canny':

        frames = [transfer_canny_image(frame.numpy()) for frame in frames]
        frames = [torch.from_numpy(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)
    elif key == 'blur':

        frames = [transfer_blur_image(frame.numpy()) for frame in frames]
        frames = [torch.from_numpy(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)
    elif key == 'lr':

        frames = [transfer_lr_image(frame.numpy()) for frame in frames]
        frames = [torch.from_numpy(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)
    elif key == 'canny_flip':

        frames = [transfer_canny_flip_image(frame.numpy()) for frame in frames]
        frames = [torch.from_numpy(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)
    elif key == 'd_m1':
        frames = transfer_depth_color_map(frames.numpy()[...,0],type = 'inferno')
    elif key == 'd_m2':
        frames = transfer_depth_color_map(frames.numpy()[...,0],type = 'viridis')
    elif key == 'd_m3':
        frames = transfer_depth_color_map(frames.numpy()[...,0],type = 'plasma')
    elif key == 'd_m4':
        frames = transfer_depth_color_map(frames.numpy()[...,0],type = 'cividis')


    if video_num_frames < max_num_frames:

        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        return frames.float().permute(0, 3, 1, 2).contiguous()
    else:

        frames = frames[:max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames


def preprocess_video_with_buckets(
    video_path: Path,
    resolution_buckets: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """
    Args:
        video_path: Path to the video file.
        resolution_buckets: List of tuples (num_frames, height, width) representing
            available resolution buckets.

    Returns:
        torch.Tensor: Video tensor with shape [F, C, H, W] where:
            F = number of frames
            C = number of channels (3 for RGB)
            H = height
            W = width

    The function processes the video through these steps:
        1. Finds nearest frame bucket <= video frame count
        2. Downsamples frames evenly to match bucket size
        3. Finds nearest resolution bucket based on dimensions
        4. Resizes frames to match bucket resolution
    """
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    resolution_buckets = [bucket for bucket in resolution_buckets if bucket[0] <= video_num_frames]
    if len(resolution_buckets) == 0:
        raise ValueError(f"video frame count in {video_path} is less than all frame buckets {resolution_buckets}")

    nearest_frame_bucket = min(
        resolution_buckets,
        key=lambda bucket: video_num_frames - bucket[0],
        default=1,
    )[0]
    frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))
    frames = video_reader.get_batch(frame_indices)
    frames = frames[:nearest_frame_bucket].float()
    frames = frames.permute(0, 3, 1, 2).contiguous()

    nearest_res = min(resolution_buckets, key=lambda x: abs(x[1] - frames.shape[2]) + abs(x[2] - frames.shape[3]))
    nearest_res = (nearest_res[1], nearest_res[2])
    frames = torch.stack([resize(f, nearest_res) for f in frames], dim=0)

    return frames
