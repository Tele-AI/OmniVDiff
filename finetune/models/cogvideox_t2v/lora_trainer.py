from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register
from einops import rearrange

class CogVideoXT2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def load_components(self,task_keys, mmc_args) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        kwargs = {
            "task_keys": task_keys,
            "mmc_args": mmc_args,
        }
        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer", **kwargs) # Construct the Transformer model and initialize its parameters based on the given model configuration.

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipeline:
        pipe = CogVideoXPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor: 
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding


    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]: # batch_size>1, merge data
        ret = {"encoded_modals":{},"prompt_embedding": [],'video_path':[]}
        
        for sample in samples:
            for k,v in sample['encoded_modals'].items():
                if k not in ret['encoded_modals'].keys(): 
                    ret['encoded_modals'][k] = []
                ret["encoded_modals"][k].append(v)

            prompt_embedding = sample["prompt_embedding"]
            ret["prompt_embedding"].append(prompt_embedding)
            ret['video_path'].append(sample['video_path'])


        for k in ret['encoded_modals'].keys():
            ret['encoded_modals'][k] = torch.stack(ret['encoded_modals'][k])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        
        return ret

    @override
    def compute_loss(self, batch) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        ### args
        use_decouple_modal = self.args.use_decouple_modal
        use_modal_emb_condgen = self.args.use_modal_emb_condgen
           
        prompt_embedding = batch["prompt_embedding"]
        latents = []
        for _,value in batch['encoded_modals'].items():
            latents.append(value)
        
        latent = torch.cat(latents, dim=1) # [B, C*3, F, H, W]


        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, C_txt_embed = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype) # torch.Size([1, 226, 4096])

        
        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # Add noise to latent
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W] # origin torch.Size([1, 16, 12, 32, 32])
        noise = torch.randn_like(latent) # [B,F,2*C,H,W]

        latent_added_noise, dict_info_decouple = self.components.scheduler.add_noise(latent, noise, timesteps, self.args, is_training = True) 
        # mask_loss_valid_modal, idx_role_modality, mask_modal_condgen_emb

        if use_modal_emb_condgen:
            ## TODO: add this at inference stage
            mask_modal_condgen_emb = dict_info_decouple['mask_modal_condgen_emb']
            emb_modal_condgen = self.components.transformer.emb_modal_condgen(mask_modal_condgen_emb)
            emb_modal_condgen = rearrange(emb_modal_condgen, 'b t c -> b 1 (t c) 1 1 ')
            latent_added_noise += emb_modal_condgen  # 加的全0

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        predicted_noise = self.components.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0] # mmv: torch.Size([1, 12, 32, 32, 32])

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_added_noise, timesteps) # mmv: torch.Size([1, 12, 32, 32, 32])

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)


        loss = weights * (latent_pred - latent) ** 2

        lambda_keys = {'rgb':1.0,'depth':1.0,'canny':1.0, \
                       'canny_flip':1.0,'segment':1.0,'depth_r':1.0, \
                       'blur': 1.0, 'lr':1.0, 'rays':1.0, \
                       'normal': 1.0, 'diffuse': 1.0, 'roughness': 1.0, \
                       'metallic': 1.0, 'f_flow': 1.0, 'b_flow': 1.0, \
                       'irradiance': 1.0, \
                       'd_m1':1.0, 'd_m2':1.0, 'd_m3':1.0, 'd_m4':1.0, \
                       'albedo':1.0, 'optic_flow': 1.0, \
                       } # weight   

        chunk_loss = torch.chunk(loss, len(batch['encoded_modals']), dim = 2) # separete loss
        if use_decouple_modal:
            mask_loss_valid_modal = dict_info_decouple['mask_loss_valid_modal'] # [B, F, C, H, W]
            B, T, C, H, W = chunk_loss[0].shape
            mask_loss_valid_modal = mask_loss_valid_modal.repeat(1, T, 1, H, W)

            chunk_mask_loss = torch.chunk(mask_loss_valid_modal, len(batch['encoded_modals']), dim = 2) 

        loss = 0.0; dict_log = {}
        invalid = 0
        for  i in range(len(batch['encoded_modals'])):
            key = list(batch['encoded_modals'].keys())[i]
            loss_pred_i = chunk_loss[i] 
            if use_decouple_modal:
                mask_loss_i = chunk_mask_loss[i]
                loss_pred_i = loss_pred_i*mask_loss_i

                n_elem = torch.sum(mask_loss_i).item()
                if n_elem==0:
                    continue

                loss_i = lambda_keys[key] * loss_pred_i.sum()/n_elem
            else:
                loss_i = (lambda_keys[key] * torch.mean(loss_pred_i.reshape(batch_size, -1), dim=1)).mean()
            

            dict_log[f'loss/loss_{key}'] = loss_i.item()

            # !!! db: If this loss term becomes NaN or Inf, reset it to 0. There might be a better solution.
            if torch.isnan(loss_i).sum()>0 or torch.isinf(loss_i).sum()>0:
                print(f"[WARNING] loss_i has NaN or inf:{key}")
                dict_log[f'loss/loss_{key}'] = 1e-6
                loss_i = 1e-6
                invalid += 1
                continue 
            loss += loss_i

        loss /= (len(batch['encoded_modals']) - invalid + 1e-6) 
        return loss, dict_log
    
    from typing import Literal, Optional
    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipeline, task_keys: List[str] = ["rgb"],
        mmc_args: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]


        
        video_list = pipe( 
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            generator=self.state.generator,
            task_keys=task_keys,
            mmc_args=mmc_args,
        )
        
        return video_list

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-t2v", "lora", CogVideoXT2VLoraTrainer)
