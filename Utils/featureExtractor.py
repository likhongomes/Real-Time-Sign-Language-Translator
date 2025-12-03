#Feature extraction using MediaPipe's hand landmarker
#Extracts hand landmarks from video frames

import os 
import cv2
import numpy as np 
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Tuple, Optional, Dict
import pickle
from pathlib import Path 
from tqdm import tqdm 
import warnings 

from Utils.config import DataConfig

class HandLandmarkExtractor:
    #MediaPipe hand landmarker provides 21 3d landmarks per hand

    LANDMARK_NAMES = [
        'WRIST',
        'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
        'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
        'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
        'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
        'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
    ]

    def __init__(
            self,
            model_path: str = "hand_landmarker.task",
            num_hands: int = 2,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5
    ):
        
        """
        Initialize the hand landmark extractor.
        
        Args:
            model_path: path to mediapipe hand landmarker model
            num_hands: maximum number of hands to detect
            min_detection_confidence: minimum confidence for detection
            min_tracking_confidence: minimum confidence for tracking
        """

        self.num_hands = num_hands
        self.model_path = model_path

        if not os.path.exists(model_path):
            self._download_model()

        #creating the detector
        base_options = python.BaseOptions(model_asset_path = model_path)
        options = vision.HandLandmarkerOptions(
            base_options = base_options,
            num_hands = num_hands,
            min_hand_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _download_model(self):
        #Downloading MediaPipe hand landmarker
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        print(f"Downloading hand landmarker model to {self.model_path}...")
        urllib.request.urlretrieve(url, self.model_path)
        print("Download is complete")

    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """ 
        Extracting hand landmarks from a single frame 
        
        Args:
            frame: BGR image from OpenCV
        Returns:
        landmarks: Array of shape (num_hands, 21, 3)
        """

        #Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Creating the MediaPipe image
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)

        #Detect hands
        result = self.detector.detect(mp_image)

        #Initializing the output array
        landmarks = np.zeros((self.num_hands, 21, 3), dtype = np.float32)

        #Extracting landmarks from each detected hand
        for hand_idx, hand_landmarks in enumerate(result.hand_landmarks[:self.num_hands]):
            for lm_idx, landmark in enumerate(hand_landmarks):
                landmarks[hand_idx, lm_idx] = [landmark.x, landmark.y, landmark.z]

        return landmarks   
    
    def extract_from_video(
            self,
            video_path: str,
            max_frames: int = 60,
            frame_skip: int = 1
    ) -> Tuple[np.ndarray, int]:
        """ 
        Extracting the hand landmarks from all frames of the video
        Args:
            video_path: path to the video file
            max_frames: maximum number of frames that we extract
            frame_skip: sample every nth frame
            
        Returns:
            landmarks: array of shape (num_frames, num_hands * 21 * 3)
            actual_frames: number of frames actually extracted 
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            warnings.warn(f"Could not open video: {video_path}")
            return np.zeros((max_frames, self.num_hands * 21 * 3), dtype = np.float32),0
        
        landmarks_sequence = []
        frame_count = 0

        while cap.isOpened() and len(landmarks_sequence) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            #skipping frames if specified 
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue 

            #Extracting landmarks
            frame_landmarks = self.extract_from_frame(frame) 

            #Flatten: (num_hands, 21, 3) -> (num_hands * 21 * 3)
            flat_landmarks = frame_landmarks.flatten()
            landmarks_sequence.append(flat_landmarks)

            frame_count += 1

        cap.release()

        actual_frames = len(landmarks_sequence)

        if actual_frames == 0:
            return np.zeros((max_frames, self.num_hands * 21 * 3), dtype = np.float32), 0
        
        #Convering to numpy array 
        landmarks_array = np.array(landmarks_sequence, dtype = np.float32)

        #padding or truncating to max_frames
        if actual_frames < max_frames:
            padding = np.zeros((max_frames - actual_frames, landmarks_array.shape[1]), dtype = np.float32)
            landmarks_array = np.vstack([landmarks_array,padding])
        elif actual_frames > max_frames:
            landmarks_array = landmarks_array[:max_frames]
            actual_frames = max_frames

        return landmarks_array, actual_frames
    
    def extract_with_augmentation(
            self,
            frame: np.ndarray,
            augment: bool = True
    ) -> np.ndarray: 
        """ 
        Extracting the landmarks with optional spatial augmentation
        
        Augmentations applied to the landmarks (not the actual images):
            random scaling
            random translation
            random rotation
            random noise 
        """

        landmarks = self.extract_from_frame(frame)

        if augment and np.random.random() < 0.5:
            landmarks = self._augment_landmarks(landmarks)
        
        return landmarks
    
    def _augment_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        #Applying the augmentation to each landmark

        augmented = landmarks.copy()

        #Random scaling (0.9 to 1.1)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            augmented[:, :, :2] *= scale 

        #Random translation
        if np.random.random() < 0.5:
            tx = np.random.uniform(-0.1, 0.1)
            ty = np.random.uniform(-0.1, 0.1)
            augmented[:, :, 0] += tx
            augmented[:, :, 1] += ty 

        #Random noise 
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented += noise.astype(np.float32)

        return augmented
    
class FeatureCache:
    #Cache extracting features to disk for faster training time 

    def __init__(self, cache_dir: str = "landmark_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents = True, exist_ok = True)

    def _get_cache_path(self, video_path: str) -> Path:
        video_name = Path(video_path).stem
        return self.cache_dir / f"{video_name}.pkl"
    
    def exists(self, video_path: str) -> bool:
        return self._get_cache_path(video_path).exists()
    
    def save(self, video_path: str, landmarks: np.ndarray, num_frames: int):
        cache_path = self._get_cache_path(video_path)
        with open(cache_path, 'wb') as f:
            pickle.dump({'landmarks': landmarks, 'num_frames': num_frames}, f)

    def load(self, video_path: str) -> Tuple[np.ndarray, int]:
        cache_path = self._get_cache_path(video_path)
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['landmarks'], data['num_frames']
    
def preprocess_dataset(
        data_dir: str,
        cache_dir: str, 
        max_frames: int = 60,
        frame_skip: int = 1,
        num_hands: int = 2
):
    """ 
    Preprocessing the entire dataset by extracting the landmarks and caching them
    Args: 
        data_dir: data directory containing video folders organized by the labels
        cache_dir: directory to save cached features
        max_frames: maximum frames per video
        frame_skip: frame sampling rate
        num_hands: number of hands to detect
    """

    extractor = HandLandmarkExtractor(num_hands = num_hands)
    cache = FeatureCache(cache_dir)

    #Getting all of the video files
    data_path = Path(data_dir) 
    video_files = list(data_path.glob("**/*.mp4"))

    print(f" Found {len(video_files)} videos to process")

    for video_path in tqdm(video_files, desc = "Extracting landmarks"):
        video_str = str(video_path)

        if cache.exists(video_str):
            continue 

        try:
            landmarks, num_frames = extractor.extract_from_video(
                video_str,
                max_frames=max_frames,
                frame_skip=frame_skip
            )
            cache.save(video_str, landmarks, num_frames)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue 

    print("Completed preprocessing")

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """ 
    Normalizing landmarks to be translation and scale invariatn
    Uses the wrist as the origin and normalizes by hand size 
    Args:
        landmarks: Shape (seq_len, num_hands * 21 *3)
    Returns:
        Normalized landmarks of same shape
    """

    seq_len, feat_dim = landmarks.shape
    num_hands = feat_dim // (21 * 3)

    normalized = landmarks.copy() 

    for t in range(seq_len):
        for h in range(num_hands):
            start_idx = h * 21 * 3
            hand_landmarks = normalized[t, start_idx:start_idx + 21 * 3].reshape(21, 3)

            #Skipping if hands not detected 
            if np.sum(np.abs(hand_landmarks)) < 1e-6:
                continue 

            #Centering on wrist 
            wrist = hand_landmarks[0].copy()
            hand_landmarks -= wrist 

            #Scaling by hand size (distance from the wrist to the middle finger tip)
            hand_size = np.linalg.norm(hand_landmarks[12])
            if hand_size > 1e-6:
                hand_landmarks /= hand_size

            normalized[t, start_idx:start_idx + 21 * 3] = hand_landmarks.flatten() 

    return normalized

if __name__ == "__main__":
    config = DataConfig() 

    print("Testing HandLandmarkExtractor...")
    extractor = HandLandmarkExtractor(num_hands=config.num_hands)
    
    # Test with a sample video if it exists
    sample_dir = Path(config.data_dir)
    if sample_dir.exists():
        video_files = list(sample_dir.glob("**/*.mp4"))
        if video_files:
            test_video = str(video_files[0])
            print(f"Testing with video: {test_video}")
            
            landmarks, num_frames = extractor.extract_from_video(test_video)
            print(f"Extracted landmarks shape: {landmarks.shape}")
            print(f"Actual frames: {num_frames}")
            
            # Test normalization
            normalized = normalize_landmarks(landmarks)
            print(f"Normalized landmarks shape: {normalized.shape}")
    else:
        print(f"Data directory {config.data_dir} not found. Run download.py first.")
