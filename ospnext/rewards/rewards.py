from PIL import Image
import numpy as np
import torch


def videoalign_score(device, load_from_pretrained=None, reward_dim="Overall", use_norm=True,
                     videoalign_root=None, save_fps=8, num_frames=None, fps=None, max_pixels=None):
    """
    VideoAlign video quality reward.
    Evaluates videos on VQ (Visual Quality), MQ (Motion Quality), TA (Text Alignment).
    Returns the specified dimension score (default: Overall = VQ + MQ + TA).

    Args:
        device: Device string (e.g. "cuda:0")
        load_from_pretrained: Path to VideoAlign checkpoint directory
        reward_dim: Which dimension to return ("VQ", "MQ", "TA", "Overall")
        use_norm: Whether to normalize scores
        videoalign_root: Path to VideoAlign source code root
        save_fps: FPS for writing temp video files
        num_frames: Number of frames for VideoAlign sampling
        fps: FPS for VideoAlign video sampling
        max_pixels: Max pixels for VideoAlign
    """
    from ospnext.rewards.videoalign_scorer import VideoAlignScorer

    scorer = VideoAlignScorer(
        device=device,
        load_from_pretrained=load_from_pretrained,
        reward_dim=reward_dim,
        use_norm=use_norm,
        save_fps=save_fps,
        num_frames=num_frames,
        fps=fps,
        max_pixels=max_pixels,
    )

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            elif images.dim() == 5 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 4, 1)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif isinstance(images, np.ndarray):
            if images.ndim == 4 and images.shape[1] == 3:
                images = images.transpose(0, 2, 3, 1)
            elif images.ndim == 5 and images.shape[2] == 3:
                images = images.transpose(0, 1, 3, 4, 2)
            elif images.ndim == 5 and images.shape[1] == 3:
                images = images.transpose(0, 2, 3, 4, 1)
            if images.dtype != np.uint8:
                images = np.clip(images * 255, 0, 255).astype(np.uint8)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "videoalign": videoalign_score,
    }
    import sys
    score_fns = {}
    for score_name, weight in score_dict.items():
        if score_name not in score_functions:
            raise ValueError(
                f"[multi_score] Unsupported score '{score_name}'. "
                f"Only videoalign-related rewards are available: {list(score_functions.keys())}"
            )
        print(f"[multi_score] Initializing score function: {score_name} (weight={weight}) ...", flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        try:
            score_fns[score_name] = (
                score_functions[score_name](device)
                if 'device' in score_functions[score_name].__code__.co_varnames
                else score_functions[score_name]()
            )
            print(f"[multi_score] Score function {score_name} initialized successfully.", flush=True)
            sys.stdout.flush(); sys.stderr.flush()
        except Exception as e:
            print(f"[multi_score] ERROR initializing {score_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            raise

    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details['avg'] = total_scores
        return score_details, {}

    return _fn


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts = [
        'A astronaut floating in zero-g',
    ]
    metadata = {}
    score_dict = {
        "videoalign": 1.0
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    scores, _ = scoring_fn(images, prompts, metadata)
    print("Scores:", scores)


if __name__ == "__main__":
    main()
