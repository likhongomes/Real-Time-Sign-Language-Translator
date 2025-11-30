import os
import shutil
import json
import yt_dlp
from moviepy.editor import VideoFileClip

# where to save
SAVE_PATH = "MS-ASL-Train"
temp_path = SAVE_PATH + "/untrimmed_videos"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

try:
    train_json = open('../MS-ASL/MSASL_train.json')
    videos = json.load(train_json)
    train_json.close()
except Exception as e:
    print(f"❌ Connection Error: {type(e).__name__}: {e}")
    exit(1)

# loop through the videos in the dataset
num_videos = len(videos)
for i in range(num_videos):
    try:
        # setup reads info from json and creates groups videos by name
        url = videos[i]['url']
        start_time = videos[i]['start_time']
        end_time = videos[i]['end_time']
        label = videos[i]['label']
        pretitle = videos[i]['clean_text']
        video_title = pretitle + str(i)
        output_title = video_title + ".mp4"
        temp_file_path = os.path.join(temp_path, video_title + ".mkv")

        folder_path = os.path.join(SAVE_PATH, pretitle)
        output_path = os.path.join(folder_path, output_title)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(f"[{i+1}/{num_videos}] {video_title}...", end=" ")

        # Download with yt-dlp (handles private videos better)
        ydl_opts = {
            'format': 'bv*+ba/b',
            'outtmpl': temp_file_path.replace('.mp4', ''),
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        print("✅ Downloaded", end=" ")

        # Trim and save
        clip = VideoFileClip(temp_file_path).subclip(start_time, end_time)
        clip.write_videofile(output_path, verbose=False, logger=None)
        clip.close()

        print("✅ Trimmed")

    except Exception as e:
        print(f"❌ Error on video {i}: {type(e).__name__}: {e}")
        continue