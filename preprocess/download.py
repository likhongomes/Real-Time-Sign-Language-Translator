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
) -> None:
    """
    Download MS-ASL videos, trim to annotated segments, extract landmarks,
    cache them, and delete the video files to keep disk footprint light.

    Idempotent: if cache already exists for a sample, it is skipped.
    """

    if data_config is None:
        data_config = DataConfig()

    SAVE_PATH = data_config.data_dir
    TEMP_PATH = os.path.join(SAVE_PATH, "tmp_videos")
    CACHE_DIR = data_config.cache_dir

    annotations_path = os.path.join(os.path.dirname(__file__), "MSASL_train.json")

    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(TEMP_PATH, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # extractor and cache
    extractor = HandLandmarkExtractor(num_hands=data_config.num_hands)
    cache = FeatureCache(CACHE_DIR)

    # load annotations
    try:
        with open(annotations_path, "r") as train_json:
            videos = json.load(train_json)
    except Exception as e:
        print(f"❌ Connection Error: {type(e).__name__}: {e}")
        print(f"Expected annotations at: {annotations_path}")
        raise SystemExit(1)

    num_videos_all = len(videos)
    if max_videos is None:
        num_videos = num_videos_all
    else:
        num_videos = min(max_videos, num_videos_all)

    print(f"Found {num_videos_all} annotated videos, processing {num_videos} of them.")

    # Main loop: download → trim → extract landmarks → cache → delete
    for i in range(num_videos):
        try:
            url = videos[i]["url"]
            start_time = videos[i]["start_time"]
            end_time = videos[i]["end_time"]
            label = videos[i]["label"]
            pretitle = videos[i]["clean_text"]
            video_title = f"{pretitle}{i}"
            output_title = f"{video_title}.mp4"

            #temporary download path
            temp_file_path = os.path.join(TEMP_PATH, f"{video_title}.mkv")

            # This folder / filename pattern MUST match what ASLDataset expects:
            # data_dir / clean_text / f"{clean_text}{i}.mp4"
            folder_path = os.path.join(SAVE_PATH, pretitle)
            output_path = os.path.join(folder_path, output_title)

            # check label folder exists
            os.makedirs(folder_path, exist_ok=True)

            print(f"[{i + 1}/{num_videos}] {video_title}...", end=" ")

            # if cache already exists then skip
            if cache.exists(output_path):
                print("✅ Cache exists, skipping download/extract")
                continue

            #download with yt-dlp
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

            #trimming and writing to output for compression
            clip = VideoFileClip(temp_file_path).subclip(start_time, end_time)
            clip.write_videofile(output_path, verbose=False, logger=None)
            clip.close()

            print("✅ Trimmed", end=" ")

            #extract landmarks and cache 'em
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

            # Delete videos to keep disk footprint light
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                print("✅ Cleaned up")
            except Exception as cleanup_err:
                print(f"⚠️  Cleanup error: {type(cleanup_err).__name__}: {cleanup_err}")

        except Exception as e:
            print(f"❌ Error on video {i}: {type(e).__name__}: {e}")
            continue

    print("\nAll videos processed.")
    print(f"Landmarks are cached in: {CACHE_DIR}")
    print("You can now train using only the cached features (no videos needed).")


if __name__ == "__main__":
    
    build_cache()
