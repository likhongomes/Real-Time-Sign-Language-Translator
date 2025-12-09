import os
import json
from typing import Any, Dict, Optional, cast
import yt_dlp
from moviepy.editor import VideoFileClip

from Utils.config import DataConfig
from Utils.featureExtractor import HandLandmarkExtractor, FeatureCache


def build_cache(
    data_config: Optional[DataConfig] = None,
    max_videos: Optional[int] = None,
    enabled: bool = True,
) -> None:
    """
    Download MS-ASL videos, trim to annotated segments, extract landmarks,
    cache them, and delete the video files to keep disk footprint light.

    - Processes train, val, and test JSONs.
    - Idempotent: if cache already exists for a sample, it is skipped.
    - `max_videos` only applies to the *train* split; val/test are always fully processed.
    - If `enabled` is False, does nothing (intended for a '--no_cache' CLI flag).
    """

    if not enabled:
        print("Cache building disabled ('no cache' requested). Skipping.")
        return

    if data_config is None:
        data_config = DataConfig()

    SAVE_PATH = data_config.data_dir
    TEMP_PATH = os.path.join(SAVE_PATH, "tmp_videos")
    CACHE_DIR = data_config.cache_dir

    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(TEMP_PATH, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # extractor and cache
    extractor = HandLandmarkExtractor(num_hands=data_config.num_hands)
    cache = FeatureCache(CACHE_DIR)

    print("=== Step 1: Download / build feature cache ===")
    print(f"data_dir  = {SAVE_PATH}")
    print(f"cache_dir = {CACHE_DIR}")

    # All three splits we care about
    split_files = [
        ("train", "MSASL_train.json"),
        ("val",   "MSASL_val.json"),
        ("test",  "MSASL_test.json"),
    ]

    for split_name, fname in split_files:
        annotations_path = os.path.join(os.path.dirname(__file__), fname)

        if not os.path.exists(annotations_path):
            print(f"[{split_name}] Annotations not found, skipping: {annotations_path}")
            continue

        # Load annotations for this split
        try:
            with open(annotations_path, "r") as f:
                videos = json.load(f)
        except Exception as e:
            print(f"[{split_name}] ❌ Failed to load {fname}: {type(e).__name__}: {e}")
            continue

        num_videos_all = len(videos)

        # Only gate the *train* split with max_videos; val/test always full
        if split_name == "train" and max_videos is not None:
            # interpret max_videos <= 0 as "don't process any new train videos"
            if max_videos <= 0:
                num_videos = 0
            else:
                num_videos = min(max_videos, num_videos_all)
        else:
            num_videos = num_videos_all

        print(
            f"\n[{split_name.upper()}] Found {num_videos_all} annotated videos, "
            f"processing {num_videos} of them."
        )

        # Main loop: download → trim → extract landmarks → cache → delete
        for i in range(num_videos):
            video_info = videos[i]
            try:
                url = video_info["url"]
                start_time = video_info["start_time"]
                end_time = video_info["end_time"]
                # MS-ASL has both "label" and "clean_text"; dataset uses clean_text as folder name
                label = video_info.get("label", "")
                pretitle = video_info["clean_text"]

                video_title = f"{pretitle}{i}"
                output_title = f"{video_title}.mp4"

                # temporary download path (.mkv from yt-dlp)
                temp_file_path = os.path.join(TEMP_PATH, f"{video_title}.mkv")

                # Folder / filename pattern MUST match what ASLDataset expects:
                # data_dir / clean_text / f"{clean_text}{i}.mp4"
                folder_path = os.path.join(SAVE_PATH, pretitle)
                output_path = os.path.join(folder_path, output_title)

                # Ensure label/clean_text folder exists
                os.makedirs(folder_path, exist_ok=True)

                print(f"[{split_name}] [{i + 1}/{num_videos}] {video_title}...", end=" ")

                # if cache already exists then skip everything
                if cache.exists(output_path):
                    print("✅ Cache exists, skipping download/extract")
                    continue

                # download with yt-dlp
                ydl_opts: Dict[str, Any] = {
                    "format": "bv*+ba/b",
                    "outtmpl": os.path.splitext(temp_file_path)[0],
                    "quiet": True,
                    "no_warnings": True,
                    "socket_timeout": 30,
                }

                with yt_dlp.YoutubeDL(cast(dict[str, object], ydl_opts)) as ydl:
                    ydl.download([url])

                print("✅ Downloaded", end=" ")

                # trim and write to final mp4
                clip = VideoFileClip(temp_file_path).subclip(start_time, end_time)
                clip.write_videofile(output_path, verbose=False, logger=None)
                clip.close()

                print("✅ Trimmed", end=" ")

                # extract landmarks and cache them
                landmarks, num_frames = extractor.extract_from_video(
                    output_path,
                    max_frames=data_config.max_frames,
                    frame_skip=data_config.frame_skip,
                )

                if num_frames == 0:
                    print("⚠️  No hands detected; skipping cache", end=" ")
                else:
                    cache.save(output_path, landmarks, num_frames)
                    print("✅ Landmarks cached", end=" ")

                # delete videos to keep disk footprint light
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    print("✅ Cleaned up")
                except Exception as cleanup_err:
                    print(
                        f"⚠️  Cleanup error: {type(cleanup_err).__name__}: {cleanup_err}"
                    )

            except Exception as e:
                print(
                    f"[{split_name}] ❌ Error on video {i}: "
                    f"{type(e).__name__}: {e}"
                )
                # keep going; don't kill the whole split on one failure
                continue

    print("\nAll splits processed.")
    print(f"Landmarks are cached in: {CACHE_DIR}")
    print("You can now train using only the cached features (no videos needed).")


if __name__ == "__main__":
    # Simple CLI behavior:
    # - `python download.py`          → build cache for all splits
    # - `python download.py no_cache` → skip cache building
    import sys

    no_cache = any(arg.lower() in {"no_cache", "no-cache", "no cache"} for arg in sys.argv[1:])
    if no_cache:
        build_cache(enabled=False)
    else:
        # No max_videos limit when run directly; adjust if you want
        build_cache()
