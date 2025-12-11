


import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import numpy as np

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME

from .utils import load_prompts, load_videos, preprocess_video_with_buckets, preprocess_video_with_resize


import json

if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)

import PIL.Image as Image
import cv2

class BaseT2VDataset(Dataset):
    """
    Base dataset class for Text-to-Video (T2V) training.

    This dataset loads prompts and videos for T2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        video_depth_column: str,
        video_canny_column: str,
        device: torch.device = None,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.prompts = load_prompts(data_root / caption_column)
        self.videos = load_videos(data_root / video_column)
        self.depth_videos = load_videos(data_root / video_depth_column)
        self.canny_videos = load_videos(data_root / video_canny_column)
        self.device = device
        self.encode_video = trainer.encode_video # 编码video
        self.encode_text = trainer.encode_text
        self.trainer = trainer

        # Check if all canny_video files exist
        if any(not path.is_file() for path in self.canny_videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.depth_videos if not path.is_file())}"
            )

        # Check if all depth_video files exist
        if any(not path.is_file() for path in self.depth_videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.depth_videos if not path.is_file())}"
            )

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        # Check if number of prompts matches number of videos
        if len(self.videos) != len(self.prompts) and len(self.depth_videos) != len(self.prompts) and len(self.canny_videos) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.videos)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        prompt = self.prompts[index]
        video = self.videos[index]
        depth_video = self.depth_videos[index]
        canny_video = self.canny_videos[index]

        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root / "cache"
        video_latent_dir = cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        depth_video_latent_dir = cache_dir / "depth_video_latent" / self.trainer.args.model_name / train_resolution_str 
        canny_video_latent_dir = cache_dir / "canny_video_latent" / self.trainer.args.model_name / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"

        video_latent_dir.mkdir(parents=True, exist_ok=True)
        depth_video_latent_dir.mkdir(parents=True, exist_ok=True)
        canny_video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_video_path = video_latent_dir / (video.stem + ".safetensors")
        encoded_depth_video_path = depth_video_latent_dir / (depth_video.stem + ".safetensors")
        encoded_canny_video_path = canny_video_latent_dir / (canny_video.stem + ".safetensors")

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

        if encoded_video_path.exists():  # rgb video
            encoded_video = load_file(encoded_video_path)["encoded_video"]
        else: 
            frames = self.preprocess(video)
            frames = frames.to(self.device)
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames)
            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)
            logger.info(f"encoded_video.shape: {encoded_video.shape}", main_process_only=False)

        if encoded_depth_video_path.exists(): # depth video
            # shape of image: [C, H, W]
            encoded_depth_video = load_file(encoded_depth_video_path)["encoded_depth_video"]
        else:
            depth_frames = self.preprocess(depth_video)
            depth_frames = depth_frames.to(self.device)
            # Current shape of frames: [F, C, H, W]
            depth_frames = self.video_transform(depth_frames)
            # Convert to [B, C, F, H, W]
            depth_frames = depth_frames.unsqueeze(0)
            depth_frames = depth_frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_depth_video = self.encode_video(depth_frames)
            

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_depth_video = encoded_depth_video[0]
            encoded_depth_video = encoded_depth_video.to("cpu")
            save_file({"encoded_depth_video": encoded_depth_video}, encoded_depth_video_path)
            logger.info(f"Saved encoded depth video to {encoded_depth_video_path}", main_process_only=False)
            logger.info(f"encoded_depth_video.shape: {encoded_depth_video.shape}",main_process_only=False)

        if encoded_canny_video_path.exists(): # canny video
            # shape of image: [C, H, W]
            encoded_canny_video = load_file(encoded_canny_video_path)["encoded_canny_video"]
        else:
            canny_frames = self.preprocess(canny_video)
            canny_frames = canny_frames.to(self.device)
            # Current shape of frames: [F, C, H, W]
            canny_frames = self.video_transform(canny_frames)
            # Convert to [B, C, F, H, W]
            canny_frames = canny_frames.unsqueeze(0)
            canny_frames = canny_frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_canny_video = self.encode_video(canny_frames)
            

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_canny_video = encoded_canny_video[0]
            encoded_canny_video = encoded_canny_video.to("cpu")
            save_file({"encoded_canny_video": encoded_canny_video}, encoded_canny_video_path)
            logger.info(f"Saved encoded canny video to {encoded_canny_video_path}", main_process_only=False)
            logger.info(f"encoded_canny_video.shape: {encoded_canny_video.shape}",main_process_only=False)


        # shape of encoded_video: [C, F, H, W]
        return {
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "encoded_depth_video": encoded_depth_video,
            "encoded_canny_video": encoded_canny_video,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
            "depth_video_metadata": {
                "num_frames": encoded_depth_video.shape[1],
                "height": encoded_depth_video.shape[2],
                "width": encoded_depth_video.shape[3],
            },
            "canny_video_metadata": {
                "num_frames": encoded_canny_video.shape[1],
                "height": encoded_canny_video.shape[2],
                "width": encoded_canny_video.shape[3],
            }
        }

    def preprocess(self, video_path: Path) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        raise NotImplementedError("Subclass must implement this method")


class BaseMMVDataset(Dataset):
    """
    Base dataset class for Multi-Modal-Video (MMV) training.

    This dataset loads prompts and videos for MMV training.

    Args:
        meta_data (str): Path to file containing metadata
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: Path,
        meta_data: str,
        task_keys: List[str],
        device: torch.device = None,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        
        super().__init__()

        data_root = Path(data_root)
        self.data_root = data_root
        meta_data = data_root / meta_data


        self.metas = self.prepare_metadata([meta_data]) 

        self.task_num = len(task_keys)

        self.task_keys = task_keys
        self.check_cache = kwargs.get("check_cache", False)
        logger.info(f"Data Initing ...")
        logger.info(f"self.task_keys: {self.task_keys}")
        # calculate modal keys
        logger.info(f"kwargs['dataset_type']: {kwargs['dataset_type']}")
        if kwargs['dataset_type'] == 'cal':
            self.cal_keys = ['canny','blur','lr','canny_flip','d_m1','d_m2','d_m3','d_m4'] # 'canny_flip','depth_r' # !!! for koala dataset
        else:
            self.cal_keys = [] 
        
        # random shuffle metas
        # for mul node,
        import time
        seed = int(time.time())
        np.random.seed(seed)
        np.random.shuffle(self.metas)

        self.modals = {}
        for task_key in self.task_keys:
            self.modals[task_key] = []
            if task_key in self.cal_keys:  
                for meta in self.metas:
                    if task_key in ['d_m1','d_m2','d_m3','d_m4']:
                        self.modals[task_key].append(Path(meta['modal']['depth'])) 
                    elif task_key in ['canny','blur','lr','canny_flip']:
                        self.modals[task_key].append(Path(meta['modal']['rgb'])) 
                continue 
            for meta in self.metas:
                self.modals[task_key].append(Path(meta['modal'][task_key]))
        self.prompts = [meta['prompt'] for meta in self.metas]

        
        self.fps = kwargs.get("gen_fps", 16) 
        self.device = device
        self.encode_video = trainer.encode_video # 编码video
        self.encode_text = trainer.encode_text
        self.trainer = trainer


        # check if all modal file exist 
        # print("[debug]*****************please cancles the comments below *******************\n\n\n\n\n\n\n\n")
        # input('input anything[y/n]')
        for key in self.task_keys:
            if any(not path.is_file() for path in self.modals[key]):
                raise ValueError(
                    f"In {key} modal. Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.modals[key] if not path.is_file())}"
                )
            
        for key in self.task_keys:
            assert len(self.modals[key]) == len(self.prompts), f"The length of {key} modals and prompts are not equal."


    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        while True:
            try:
                dict_item = self.get_one_item(index)
                return dict_item
            except Exception as e:
                logger.info(f"Error while loading item {index}: {e}")
                index = (index + 1) % len(self.prompts)
                with open("exceptions_dataset.txt", "a") as f:
                    f.write(f"Error {e}\n")
                    f.write(f"Error while loading item {index}: {e}\n")

    def get_one_item(self, index):
        cache_dir = self.data_root / "cache"
        prompt = self.prompts[index]
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        
        # 1. load prompt embedding 
        if prompt_embedding_path.exists(): 
            if not self.check_cache: 
                prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
                assert torch.is_tensor(prompt_embedding), f"Prompt embedding should be a tensor.{prompt}"
                assert torch.isnan(prompt_embedding).sum()==0, f"Prompt embedding has NaN: {torch.isnan(prompt_embedding).sum()},{prompt}"
                assert torch.isinf(prompt_embedding).sum()==0, f"Prompt embedding has inf: {torch.isinf(prompt_embedding).sum()},{prompt}"
                logger.debug(
                    f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                    main_process_only=False,
                )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)


        # different modality deal
        encoded_modals = {}
        for key in self.task_keys:
            video = self.modals[key][index]
 
            import os
            parent_dir = os.path.basename(os.path.dirname(video))
            train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
            video_latent_dir = cache_dir / f"{key}_video_latent" / self.trainer.args.model_name / train_resolution_str / parent_dir
            video_latent_dir.mkdir(parents=True, exist_ok=True)

            encoded_video_path = video_latent_dir / (video.stem + ".safetensors")
            if encoded_video_path.exists(): 
                if not self.check_cache:
                    encoded_video = load_file(encoded_video_path)["encoded_video"]
                    logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
            else:
                # preprocess video
                if key in self.cal_keys: # canny etc.
                    frames = self.preprocess(video,key,target_fps=self.fps)
                else:
                    frames = self.preprocess(video,target_fps=self.fps)

                frames = frames.to(self.device)
                # Current shape of frames: [F, C, H, W]
                frames = self.video_transform(frames)

                # Convert to [B, C, F, H, W]
                frames = frames.unsqueeze(0)
                frames = frames.permute(0, 2, 1, 3, 4).contiguous()
                encoded_video = self.encode_video(frames)

                # [1, C, F, H, W] -> [C, F, H, W]
                encoded_video = encoded_video[0]
                encoded_video = encoded_video.to("cpu")
                save_file({"encoded_video": encoded_video}, encoded_video_path)
                logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)

            if not self.check_cache:
                assert torch.is_tensor(encoded_video), f"Encoded video is not a tensor: {type(encoded_video)},path: {video}"
                assert torch.isnan(encoded_video).sum()==0, f"encoded_video has NaN: {torch.isnan(encoded_video).sum()},path: {video}"
                assert torch.isinf(encoded_video).sum()==0, f"encoded_video has inf: {torch.isinf(encoded_video).sum()},path: {video}"

                encoded_modals[key] = encoded_video

        return {
            "prompt_embedding": prompt_embedding,
            "encoded_modals": encoded_modals, 
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
            "video_path": self.modals['rgb'][index], # for debug
        } if not self.check_cache else {}


    def preprocess(self, video_path: Path,key:str=None,target_fps=-1) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        raise NotImplementedError("Subclass must implement this method")

    def prepare_metadata(self, metadatas: List[str]):
        metas = []
        cnt = 0
        for metadata in metadatas:
            print(f"[INFO] Processing {metadata}")
            with open(metadata, "r") as f:
                for line in f:
                    cnt += 1
                    metas.append(json.loads(line))
                    
        self.metas = metas
        print(f"[INFO] Total {len(self.metas)} samples")
        return self.metas

class T2VDatasetWithResize(BaseT2VDataset):
    """
    A dataset class for text-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

    @override
    def preprocess(self, video_path: Path,key:str=None) -> torch.Tensor:
        return preprocess_video_with_resize(
            video_path,
            self.max_num_frames,
            self.height,
            self.width,
            key,
        )

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)

class MMVDatasetWithResize(BaseMMVDataset):
    """
    A dataset class for text-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

    @override
    def preprocess(self, video_path: Path,key:str=None,target_fps=-1) -> torch.Tensor:
        return preprocess_video_with_resize(
            video_path,
            self.max_num_frames,
            self.height,
            self.width,
            key,
            target_fps
        )

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)

class T2VDatasetWithBuckets(BaseT2VDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        """ """
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]

        self.__frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        return preprocess_video_with_buckets(video_path, self.video_resolution_buckets)

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)

if  __name__ == "__main__":
    base_dataset = BaseMMVDataset('xxx.jsonl')