#Real-time ASL Recognition from Live Camera Feed
#Uses trained model to recognize signs from webcam input

import os
import time
import argparse
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2 #type: ignore[import, attr-defined]
from mediapipe.solutions import hands as mp_hands #type: ignore[import, attr-defined]
from mediapipe.solutions import drawing_utils as mp_drawing #type: ignore[import, attr-defined]
from mediapipe.solutions import drawing_styles as mp_drawing_styles #type: ignore[import, attr-defined]

from config import InferenceConfig, ModelConfig
from models import create_model
from featureExtractor import normalize_landmarks


class ASLInference:
    """
    Real-time ASL recognition from camera feed.
    
    Features:
    - Continuous hand landmark extraction
    - Sliding window prediction
    - Temporal smoothing for stable predictions
    - Visual feedback with landmarks overlay
    """
    
    def __init__(
        self,
        model_path: str,
        camera_id: int = 0,
        confidence_threshold: float = 0.5,
        prediction_window: int = 30,
        num_hands: int = 2,
        show_landmarks: bool = True
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            camera_id: Camera device ID
            confidence_threshold: Minimum confidence for predictions
            prediction_window: Number of frames for prediction
            num_hands: Maximum hands to detect
            show_landmarks: Whether to visualize landmarks
        """
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.prediction_window = prediction_window
        self.num_hands = num_hands
        self.show_landmarks = show_landmarks
        self.feature_dim = num_hands * 21 * 3
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model, self.label_mapping, self.idx_to_label = self._load_model(model_path)
        self.model.eval()
        
        # Setup MediaPipe hand detector
        self.mp_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Drawing utilities
        self.mp_drawing: Any = mp.drawing
        self.mp_drawing_styles: Any = mp.drawing_styles
        
        # Frame buffer for temporal predictions
        self.frame_buffer = deque(maxlen=prediction_window)
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, Dict, Dict]:
        """Load trained model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model config
        config = checkpoint.get('config', {})
        model_config_dict = config.get('model', {})
        
        model_config = ModelConfig()
        for key, value in model_config_dict.items():
            if hasattr(model_config, key):
                setattr(model_config, key)
        
        # Get label mapping
        label_mapping = checkpoint.get('label_mapping', {})
        model_config.num_classes = len(label_mapping)
        
        # Create model
        model = create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Create reverse mapping
        idx_to_label = {v: k for k, v in label_mapping.items()}
        
        print(f"Loaded model with {len(label_mapping)} classes")
        
        return model, label_mapping, idx_to_label
    
    def extract_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Extract hand landmarks from a frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        
        landmarks = np.zeros((self.num_hands, 21, 3), dtype=np.float32)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:self.num_hands]):
                for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[hand_idx, lm_idx] = [landmark.x, landmark.y, landmark.z]
        
        return landmarks.flatten()
    
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw hand landmarks on frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_frame
    
    @torch.no_grad()
    def predict(self, landmarks_sequence: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Make prediction from landmark sequence.
        
        Args:
            landmarks_sequence: Shape (seq_len, feature_dim)
            
        Returns:
            predicted_label: Top predicted sign
            confidence: Prediction confidence
            top_k: Top-5 predictions with confidences
        """
        # Normalize landmarks
        normalized = normalize_landmarks(landmarks_sequence)
        
        # Convert to tensor and add batch dimension
        x = torch.from_numpy(normalized).float().unsqueeze(0).to(self.device)
        seq_len = torch.tensor([landmarks_sequence.shape[0]], device=self.device)
        
        # Forward pass
        logits = self.model(x, seq_len)
        probs = F.softmax(logits, dim=1)
        
        # Get top-5 predictions
        top_probs, top_indices = probs.topk(5, dim=1)
        
        top_k = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = self.idx_to_label.get(idx.item(), "Unknown")
            top_k.append((label, prob.item()))
        
        predicted_label = top_k[0][0]
        confidence = top_k[0][1]
        
        return predicted_label, confidence, top_k
    
    def smooth_prediction(self, label: str, confidence: float) -> Tuple[str, float]:
        """Apply temporal smoothing to predictions"""
        self.prediction_history.append((label, confidence))
        
        if len(self.prediction_history) < 3:
            return label, confidence
        
        # Count occurrences
        label_scores = {}
        for past_label, past_conf in self.prediction_history:
            if past_label not in label_scores:
                label_scores[past_label] = []
            label_scores[past_label].append(past_conf)
        
        # Find most common with highest average confidence
        best_label = label
        best_score = 0
        
        for l, confs in label_scores.items():
            score = len(confs) * np.mean(confs)
            if score > best_score:
                best_score = score
                best_label = l
        
        avg_conf = np.mean(label_scores.get(best_label, [confidence]))
        
        return best_label, avg_conf
    
    def run(self):
        """Run real-time inference"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nASL Recognition Started!")
        print("Press 'q' to quit, 'c' to clear buffer")
        print("-" * 40)
        
        current_prediction = ""
        current_confidence = 0.0
        top_predictions = []
        
        fps_counter = 0
        fps_start = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Mirror frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                self.frame_buffer.append(landmarks)
                
                # Make prediction when buffer is full enough
                if len(self.frame_buffer) >= self.prediction_window // 2:
                    landmarks_sequence = np.array(list(self.frame_buffer))
                    
                    # Check if hands are detected
                    if np.sum(np.abs(landmarks_sequence)) > 0.1:
                        label, conf, top_k = self.predict(landmarks_sequence)
                        label, conf = self.smooth_prediction(label, conf)
                        
                        if conf >= self.confidence_threshold:
                            current_prediction = label
                            current_confidence = conf
                            top_predictions = top_k
                
                # Draw landmarks if enabled
                if self.show_landmarks:
                    frame = self.draw_landmarks(frame)
                
                # Draw prediction overlay
                frame = self._draw_overlay(
                    frame, current_prediction, current_confidence, top_predictions, fps
                )
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start = time.time()
                
                # Show frame
                cv2.imshow('ASL Recognition', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.frame_buffer.clear()
                    self.prediction_history.clear()
                    current_prediction = ""
                    current_confidence = 0.0
                    print("Buffer cleared")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mp_hands.close()
    
    def _draw_overlay(
        self,
        frame: np.ndarray,
        prediction: str,
        confidence: float,
        top_k: List[Tuple[str, float]],
        fps: int
    ) -> np.ndarray:
        """Draw prediction overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Buffer status
        buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.prediction_window}"
        cv2.putText(frame, buffer_status, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Main prediction
        if prediction:
            color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 165, 255)
            cv2.putText(frame, f"{prediction}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Top-5 predictions
        if top_k:
            cv2.putText(frame, "Top 5:", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            for i, (label, conf) in enumerate(top_k[:5]):
                text = f"{i+1}. {label}: {conf:.2%}"
                cv2.putText(frame, text, (10, 160 + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Q: Quit | C: Clear", (w - 150, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame


class ASLInferenceBatch:
    """Batch inference for videos or image sequences"""
    
    def __init__(self, model_path: str, num_hands: int = 2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_hands = num_hands
        self.feature_dim = num_hands * 21 * 3
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        model_config_dict = config.get('model', {})
        
        model_config = ModelConfig()
        for key, value in model_config_dict.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        self.label_mapping = checkpoint.get('label_mapping', {})
        model_config.num_classes = len(self.label_mapping)
        
        self.model = create_model(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Setup MediaPipe
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=num_hands,
            min_detection_confidence=0.5
        )
    
    def predict_video(self, video_path: str, max_frames: int = 60) -> Tuple[str, float]:
        """Predict sign from video file"""
        from featureExtractor import HandLandmarkExtractor, normalize_landmarks
        
        extractor = HandLandmarkExtractor(num_hands=self.num_hands)
        landmarks, num_frames = extractor.extract_from_video(video_path, max_frames=max_frames)
        
        if num_frames == 0:
            return "No hands detected", 0.0
        
        # Normalize and predict
        normalized = normalize_landmarks(landmarks)
        x = torch.from_numpy(normalized).float().unsqueeze(0).to(self.device)
        seq_len = torch.tensor([num_frames], device=self.device)
        
        with torch.no_grad():
            logits = self.model(x, seq_len)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        
        label = self.idx_to_label.get(idx.item(), "Unknown")
        return label, conf.item()
    
    def close(self):
        self.mp_hands.close()


def main():
    parser = argparse.ArgumentParser(description='Real-time ASL Recognition')
    
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--window', type=int, default=30,
                        help='Prediction window size')
    parser.add_argument('--no_landmarks', action='store_true',
                        help='Disable landmark visualization')
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = ASLInference(
        model_path=args.model,
        camera_id=args.camera,
        confidence_threshold=args.threshold,
        prediction_window=args.window,
        show_landmarks=not args.no_landmarks
    )
    
    # Run
    inference.run()


if __name__ == "__main__":
    main()