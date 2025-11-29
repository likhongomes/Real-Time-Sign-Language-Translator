import os
import subprocess
from urllib.parse import urlparse, parse_qs

# from pytube import YouTube

# def video_id_from_url(url):
#     if not url.startswith("http"):
#         url = "https://" + url  # handle 'www.youtube.com/...'
#     qs = parse_qs(urlparse(url).query)
#     return qs.get("v", [None])[0]

# def download_youtube(url, out_dir="msasl/raw_videos"):
#     os.makedirs(out_dir, exist_ok=True)
#     vid = video_id_from_url(url)
#     if vid is None:
#         raise ValueError(f"Cannot parse video id from {url}")
#     out_path = os.path.join(out_dir, f"{vid}.mp4")
#     if os.path.exists(out_path):
#         return out_path  # already downloaded

#     # requires `yt-dlp` installed: pip install yt-dlp
#     cmd = [
#         "yt-dlp",
#         "-f", "mp4",
#         "-o", out_path,
#         url
#     ]
#     subprocess.run(cmd, check=True)
#     return out_path

# download_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

import yt_dlp
import shutil


def download_video(url: str, out_dir: str = "downloads"):
    os.makedirs(out_dir, exist_ok=True)

    # Check if ffmpeg is available. yt-dlp needs ffmpeg to merge separate
    # video/audio streams (e.g. bestvideo+bestaudio). If ffmpeg is missing,
    # request a single-file format to avoid aborting with the error you saw.
    ffmpeg_installed = shutil.which("ffmpeg") is not None

    if ffmpeg_installed:
        format_spec = "bv*+ba/best"  # request best video+audio (may require merging)
    else:
        print(
            "Warning: ffmpeg not found in PATH. Falling back to a single-file format to avoid merging."
        )
        # prefer mp4 single-file when available to avoid requiring ffmpeg
        format_spec = "best[ext=mp4]/best"

    # Options for yt-dlp
    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "format": format_spec,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    download_video(video_url)
