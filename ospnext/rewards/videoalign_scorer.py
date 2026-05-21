import os
import tempfile
import shutil
import numpy as np
import torch
import json
import argparse
from typing import List, Union, Optional
from PIL import Image

class VideoAlignScorer:
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_from_pretrained: str = None,
        load_from_pretrained_step: int = -1,
        reward_dim: str = "Overall",  # "VQ", "MQ", "TA", "Overall"
        use_norm: bool = True,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        max_pixels: Optional[int] = None,
        save_fps: int = 8,  # fps for writing temp video files
    ):
        self.device = device
        self.reward_dim = reward_dim
        self.use_norm = use_norm
        self.fps = fps
        self.num_frames = num_frames
        self.max_pixels = max_pixels
        self.save_fps = save_fps

        # Import from local module (no sys.path hack needed)
        from ospnext.rewards.videoalign_inference import VideoVLMRewardInference

        self.inferencer = VideoVLMRewardInference(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            device=device,
            dtype=dtype,
        )

        print(f"[VideoAlignScorer] Initialized with checkpoint: {load_from_pretrained}")
        print(f"[VideoAlignScorer] reward_dim={reward_dim}, use_norm={use_norm}")

    def _save_video_to_tempfile(self, frames: np.ndarray, tmp_dir: str, idx: int) -> str:
        from torchvision.io import write_video

        video_path = os.path.join(tmp_dir, f"tmp_video_{idx}.mp4")
        video_tensor = torch.from_numpy(frames).to(torch.uint8)
        write_video(video_path, video_tensor, self.save_fps, video_codec="h264")
        return video_path

    def _convert_to_video_frames(self, images) -> List[np.ndarray]:
        result = []
        if isinstance(images, np.ndarray):
            if images.ndim == 5:
                for i in range(images.shape[0]):
                    result.append(images[i])
            elif images.ndim == 4:
                for i in range(images.shape[0]):
                    result.append(images[i:i+1])
        elif isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy() if images.is_cuda else images.numpy()
            return self._convert_to_video_frames(images_np)
        elif isinstance(images, (list, tuple)):
            for img in images:
                if isinstance(img, np.ndarray):
                    if img.ndim == 4:
                        result.append(img)
                    elif img.ndim == 3:
                        result.append(img[np.newaxis])
                elif isinstance(img, Image.Image):
                    arr = np.array(img)
                    result.append(arr[np.newaxis])
                elif isinstance(img, torch.Tensor):
                    arr = img.cpu().numpy() if img.is_cuda else img.numpy()
                    if arr.ndim == 4:
                        result.append(arr)
                    elif arr.ndim == 3:
                        result.append(arr[np.newaxis])
        return result

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List, np.ndarray, torch.Tensor],
        prompts: List[str],
    ) -> List[float]:
        """Backward compatibility for raw frames input."""
        video_frames_list = self._convert_to_video_frames(images)
        assert len(video_frames_list) == len(prompts), \
            f"Number of videos ({len(video_frames_list)}) must match prompts ({len(prompts)})"

        tmp_dir = tempfile.mkdtemp(prefix="videoalign_")
        try:
            video_paths = []
            for i, frames in enumerate(video_frames_list):
                if frames.dtype != np.uint8:
                    frames = np.clip(frames * 255 if frames.max() <= 1.0 else frames, 0, 255).astype(np.uint8)
                path = self._save_video_to_tempfile(frames, tmp_dir, i)
                video_paths.append(path)

            return self.score_files(video_paths, prompts)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @torch.no_grad()
    def score_files(self, video_paths: List[str], prompts: List[str]) -> List[float]:
        """
        Directly compute VideoAlign reward scores for existing video files.
        """
        assert len(video_paths) == len(prompts), \
            f"Number of video files ({len(video_paths)}) must match prompts ({len(prompts)})"
            
        rewards = self.inferencer.reward(
            video_paths=video_paths,
            prompts=prompts,
            fps=self.fps,
            num_frames=self.num_frames,
            max_pixels=self.max_pixels,
            use_norm=self.use_norm,
        )

        scores = [r[self.reward_dim] for r in rewards]
        return scores


def process_video_folders(main_dir: str, prompt_file: str, scorer: VideoAlignScorer):
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt txt file not found: {prompt_file}")
        
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Loaded {len(prompts)} prompts from {prompt_file}.")

    for subdir_name in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir_name)

        if not os.path.isdir(subdir_path):
            continue
            
        video_paths = []
        batch_prompts = []
        video_names = []

        for i, prompt in enumerate(prompts):
            vid_name = f"video_{i}.mp4"
            vid_path = os.path.join(subdir_path, vid_name)
            
            if os.path.exists(vid_path):
                video_paths.append(vid_path)
                batch_prompts.append(prompt)
                video_names.append(vid_name)
                
        if not video_paths:
            print(f"No corresponding videos found in {subdir_path}, skipping.")
            continue
            
        print(f"\nProcessing {len(video_paths)} videos in {subdir_path}...")

        try:
            scores = scorer.score_files(video_paths, batch_prompts)

            results = {}
            for v_name, score in zip(video_names, scores):
                results[v_name] = float(score)
                
            avg_score = sum(scores) / len(scores)
            
            output_data = {
                "scores": results,
                "average_score": float(avg_score)
            }
            
            out_json = os.path.join(subdir_path, "videoalign_scores.json")
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
                
            print(f"Success! Saved results to {out_json} (Average Score: {avg_score:.4f})")
            
        except Exception as e:
            print(f"Error processing {subdir_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and calculate VideoAlign scores.")
    parser.add_argument("--main_dir", type=str, default="/home/ma-user/work/xianyi/osp_next/ospnext/samples/osp_next_mixgrpo/moviegen/lr104_bf16_2", help="Path to the main folder containing subfolders of videos.")
    parser.add_argument("--prompt_file", type=str, default="/home/ma-user/work/xianyi/osp_next/ospnext/assets/t2v/eval_Moviegen.txt", help="Path to the txt file containing prompts.")
    parser.add_argument("--ckpt_path", type=str, default="/home/ma-user/work/xianyi/ckpts/KlingTeam/VideoReward", help="Path to VideoAlign checkpoint.")
    parser.add_argument("--device", type=str, default="npu", help="Device to run on (e.g., cuda, npu).")
    parser.add_argument("--reward_dim", type=str, default="Overall", choices=["VQ", "MQ", "TA", "Overall"], help="Reward dimension.")
    
    args = parser.parse_args()

    scorer = VideoAlignScorer(
        device=args.device,
        load_from_pretrained=args.ckpt_path,
        reward_dim=args.reward_dim,
    )

    process_video_folders(
        main_dir=args.main_dir,
        prompt_file=args.prompt_file,
        scorer=scorer
    )