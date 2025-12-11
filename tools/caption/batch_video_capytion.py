import io

import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/xdb/huggface_model/cogvlm2-llama3-caption"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
# parser.add_argument('--video_text', type=str, help='Path to video_text file list', required=True)
args = parser.parse_args([])


def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True
).eval().to(DEVICE)


def predict(prompt, video_data, temperature):
    strategy = 'chat'

    video = load_video(video_data, strategy=strategy)

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response



def test():
    prompt = "Please describe this video in detail."
    temperature = 0.1

    video_text = "videos_rgb_PlayingGuitar.txt"
    print(f"Reading video list from: {video_text}")
    
    with open(video_text, 'r') as f:
        video_list = f.read().splitlines()

    from tqdm import tqdm

    # 打开文件用于保存所有的结果
    video_res_text = f"{video_text[:-4]}_caption.txt"
    with open(video_res_text, 'w') as f:
        # 遍历视频列表
        for video_path in tqdm(video_list):
            # print(f"Processing video: {video_path}")
            video_data = open(video_path, 'rb').read()
            response = predict(prompt, video_data, temperature)

            # 每次处理一个视频后，立即将结果写入文件
            # f.write(f"Video: {video_path}\n")
            # for item in response:
            print(f"Caption for {video_path} is: {response}")
            f.write(f"{response}")
            f.write("\n")  # 每个视频的结果分隔一行

            # print(f"Caption for {video_path} saved in: {video_res_text}")

    print(f"All captions saved in: {video_res_text}")


if __name__ == '__main__':
    test()
