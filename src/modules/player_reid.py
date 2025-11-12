"""
Player Re-Identification module.

This module helps maintain consistent player IDs even when
players temporarily leave and re-enter the frame.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..config import setup_logging

logger = setup_logging(__name__)

if FAISS_AVAILABLE:
    logger.info("Faiss library available - using optimized similarity search")
else:
    logger.warning("Faiss not available - falling back to sklearn (slower)")

if TORCH_AVAILABLE:
    logger.info("PyTorch available - MobileNetV3 feature extraction enabled")
else:
    logger.warning("PyTorch not available - using manual feature extraction")


@dataclass
class PlayerEmbedding:
    """Data class representing a player's visual embedding."""
    player_id: int
    embedding: np.ndarray
    last_seen_frame: int
    bbox: List[int]
    team: Optional[str] = None
    confidence: float = 1.0


class MobileNetV3FeatureExtractor:
    """Feature extractor using MobileNetV3 pretrained on ImageNet."""

    def __init__(self, feature_size: int = 1280):
        """
        Initialize MobileNetV3 feature extractor.

        Args:
            feature_size: Output feature dimension (1280 for MobileNetV3-Large)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("MobileNetV3 requires PyTorch. Install with: pip install torch torchvision")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_size = feature_size

        # Load MobileNetV3-Large pretrained on ImageNet
        mobilenet = models.mobilenet_v3_large(pretrained=True)

        # Remove classifier layer to get features
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"MobileNetV3 feature extractor initialized on {self.device}")

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from image using MobileNetV3.

        Args:
            image: Image as numpy array (BGR from OpenCV)

        Returns:
            Feature vector (numpy array)
        """
        if image.size == 0:
            return np.zeros(self.feature_size, dtype=np.float32)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
            features = features.squeeze()  # Remove batch and spatial dimensions
            features = features.cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features.astype(np.float32)


class PlayerReID:
    """Player re-identification using visual features with Faiss acceleration and MobileNetV3."""

    def __init__(self, feature_size: int = 128, max_gallery_size: int = 50,
                 similarity_threshold: float = 0.7, use_faiss: bool = True,
                 use_mobilenet: bool = True):
        """
        Initialize ReID module.

        Args:
            feature_size: Size of feature embeddings (128 for manual, 1280 for MobileNetV3)
            max_gallery_size: Maximum number of embeddings to store per player
            similarity_threshold: Minimum similarity for re-identification
            use_faiss: Whether to use Faiss for similarity search (if available)
            use_mobilenet: Whether to use MobileNetV3 for feature extraction (if available)
        """
        self.use_mobilenet = use_mobilenet and TORCH_AVAILABLE

        # Initialize feature extractor
        if self.use_mobilenet:
            self.mobilenet_extractor = MobileNetV3FeatureExtractor(feature_size=1280)
            self.feature_size = 1280
            logger.info("Using MobileNetV3 for feature extraction")
        else:
            self.mobilenet_extractor = None
            self.feature_size = feature_size
            logger.info("Using manual feature extraction (color + texture)")

        self.max_gallery_size = max_gallery_size
        self.similarity_threshold = similarity_threshold
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        # Gallery of known player embeddings
        self.player_gallery = {}  # {player_id: [embeddings]}

        # Mapping between temporary and persistent IDs
        self.id_mapping = {}
        self.next_persistent_id = 1

        # Faiss index for fast similarity search (Inner Product = cosine similarity for L2-normalized vectors)
        if self.use_faiss:
            self.faiss_index = faiss.IndexFlatIP(self.feature_size)  # Inner Product index
            self.faiss_id_map = []  # Maps Faiss index position to (player_id, embedding_idx)
            logger.info(f"Initialized PlayerReID with Faiss (feature_size={self.feature_size})")
        else:
            self.faiss_index = None
            self.faiss_id_map = None
            logger.info(f"Initialized PlayerReID with sklearn (feature_size={self.feature_size})")

    def extract_features(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract visual features from player crop.

        Uses MobileNetV3 if enabled, otherwise falls back to manual features
        (color histogram and texture).

        Args:
            frame: Full frame image
            bbox: Player bounding box [x1, y1, x2, y2]

        Returns:
            Feature vector (numpy array)
        """
        x1, y1, x2, y2 = bbox

        # Crop player region
        player_crop = frame[y1:y2, x1:x2]

        if player_crop.size == 0:
            return np.zeros(self.feature_size)

        # Focus on upper body (jersey area)
        h, w = player_crop.shape[:2]
        upper_crop = player_crop[:h//2, :]

        if upper_crop.size == 0:
            upper_crop = player_crop

        # Use MobileNetV3 if enabled
        if self.use_mobilenet:
            return self.mobilenet_extractor.extract(upper_crop)

        # Otherwise use manual feature extraction
        # Resize to standard size
        resized = cv2.resize(upper_crop, (64, 128))

        # Extract multiple features
        features = []

        # 1. Color histogram (HSV space)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)

        features.extend(hist_h)
        features.extend(hist_s)
        features.extend(hist_v)

        # 2. Dominant color
        pixels = resized.reshape(-1, 3)
        dominant_color = np.median(pixels, axis=0)
        dominant_color = dominant_color / 255.0

        features.extend(dominant_color)

        # 3. Texture (edge density)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        features.append(edge_density)

        # 4. Spatial color distribution (divide into grid)
        grid_size = 2
        h_step = resized.shape[0] // grid_size
        w_step = resized.shape[1] // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                grid_crop = resized[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                if grid_crop.size > 0:
                    grid_color = np.mean(grid_crop, axis=(0, 1)) / 255.0
                    features.extend(grid_color)

        # Convert to numpy array and pad/trim to feature_size
        features = np.array(features, dtype=np.float32)

        if len(features) < self.feature_size:
            # Pad with zeros
            features = np.pad(features, (0, self.feature_size - len(features)))
        elif len(features) > self.feature_size:
            # Trim
            features = features[:self.feature_size]

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def add_to_gallery(self, player_id: int, embedding: np.ndarray,
                      frame_number: int, bbox: List[int], team: str = None):
        """
        Add player embedding to gallery and Faiss index.

        Args:
            player_id: Player ID
            embedding: Feature embedding (must be L2-normalized)
            frame_number: Frame number
            bbox: Bounding box
            team: Team label
        """
        if player_id not in self.player_gallery:
            self.player_gallery[player_id] = []

        player_emb = PlayerEmbedding(
            player_id=player_id,
            embedding=embedding,
            last_seen_frame=frame_number,
            bbox=bbox,
            team=team
        )

        # Get embedding index before adding
        embedding_idx = len(self.player_gallery[player_id])

        self.player_gallery[player_id].append(player_emb)

        # Add to Faiss index
        if self.use_faiss:
            # Faiss requires float32 and shape (1, feature_size)
            emb_array = embedding.astype(np.float32).reshape(1, -1)
            self.faiss_index.add(emb_array)
            self.faiss_id_map.append((player_id, embedding_idx))

        # Limit gallery size (keep most recent)
        if len(self.player_gallery[player_id]) > self.max_gallery_size:
            self.player_gallery[player_id] = self.player_gallery[player_id][-self.max_gallery_size:]

            # Note: For simplicity, we don't remove from Faiss index
            # In production, you'd use IndexIDMap to allow removal
            # For now, old embeddings stay in index but won't match due to frame_gap check

    def find_best_match(self, query_embedding: np.ndarray,
                       frame_number: int,
                       max_frames_gap: int = 300,
                       k_neighbors: int = 10) -> Tuple[Optional[int], float]:
        """
        Find best matching player from gallery using Faiss or sklearn.

        Args:
            query_embedding: Query feature embedding (L2-normalized)
            frame_number: Current frame number
            max_frames_gap: Maximum frame gap for re-identification
            k_neighbors: Number of nearest neighbors to retrieve from Faiss

        Returns:
            Tuple of (best_match_id, similarity_score)
        """
        if not self.player_gallery:
            return None, 0.0

        if self.use_faiss:
            return self._find_best_match_faiss(query_embedding, frame_number,
                                              max_frames_gap, k_neighbors)
        else:
            return self._find_best_match_sklearn(query_embedding, frame_number,
                                                max_frames_gap)

    def _find_best_match_faiss(self, query_embedding: np.ndarray,
                              frame_number: int,
                              max_frames_gap: int = 300,
                              k_neighbors: int = 10) -> Tuple[Optional[int], float]:
        """
        Find best match using Faiss for fast similarity search.

        Args:
            query_embedding: Query feature embedding
            frame_number: Current frame number
            max_frames_gap: Maximum frame gap
            k_neighbors: Number of neighbors to search

        Returns:
            Tuple of (best_match_id, similarity_score)
        """
        if self.faiss_index.ntotal == 0:
            return None, 0.0

        # Prepare query for Faiss (needs float32 and shape (1, feature_size))
        query = query_embedding.astype(np.float32).reshape(1, -1)

        # Search k nearest neighbors
        k = min(k_neighbors, self.faiss_index.ntotal)
        similarities, indices = self.faiss_index.search(query, k)

        # Flatten results
        similarities = similarities[0]  # Shape: (k,)
        indices = indices[0]  # Shape: (k,)

        # Find best valid match
        best_match_id = None
        best_similarity = 0.0

        for sim, idx in zip(similarities, indices):
            if idx < 0 or idx >= len(self.faiss_id_map):
                continue

            player_id, emb_idx = self.faiss_id_map[idx]

            # Check if player exists and embedding is within frame gap
            if player_id not in self.player_gallery:
                continue

            if emb_idx >= len(self.player_gallery[player_id]):
                continue

            player_emb = self.player_gallery[player_id][emb_idx]

            # Check frame gap
            if frame_number - player_emb.last_seen_frame > max_frames_gap:
                continue

            # Found valid match
            if sim > best_similarity:
                best_similarity = float(sim)
                best_match_id = player_id

        # Only return match if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_id, best_similarity

        return None, best_similarity

    def _find_best_match_sklearn(self, query_embedding: np.ndarray,
                                 frame_number: int,
                                 max_frames_gap: int = 300) -> Tuple[Optional[int], float]:
        """
        Find best match using sklearn (fallback when Faiss not available).

        Args:
            query_embedding: Query feature embedding
            frame_number: Current frame number
            max_frames_gap: Maximum frame gap

        Returns:
            Tuple of (best_match_id, similarity_score)
        """
        best_match_id = None
        best_similarity = 0.0

        for player_id, embeddings in self.player_gallery.items():
            # Only consider players seen recently
            recent_embeddings = [
                emb for emb in embeddings
                if frame_number - emb.last_seen_frame <= max_frames_gap
            ]

            if not recent_embeddings:
                continue

            # Calculate similarity with all embeddings for this player
            similarities = []
            for emb in recent_embeddings:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    emb.embedding.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)

            # Use maximum similarity
            max_similarity = max(similarities)

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match_id = player_id

        # Only return match if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_id, best_similarity

        return None, best_similarity

    def update_tracking(self, frame: np.ndarray, detections: List[Dict],
                       frame_number: int) -> List[Dict]:
        """
        Update player tracking with re-identification.

        Args:
            frame: Current frame
            detections: List of player detections with temporary IDs
            frame_number: Current frame number

        Returns:
            Updated detections with persistent IDs
        """
        updated_detections = []

        for detection in detections:
            bbox = detection.get('bbox')
            temp_id = detection.get('player_id') or detection.get('track_id')
            team = detection.get('team')

            if not bbox:
                continue

            # Extract features
            embedding = self.extract_features(frame, bbox)

            # Try to find existing match
            matched_id, similarity = self.find_best_match(embedding, frame_number)

            if matched_id is not None:
                # Re-identified existing player
                persistent_id = matched_id
                logger.debug(f"Re-identified player {temp_id} as {persistent_id} "
                           f"(similarity: {similarity:.3f})")
            else:
                # Check if we've already assigned a persistent ID to this temp ID
                if temp_id in self.id_mapping:
                    persistent_id = self.id_mapping[temp_id]
                else:
                    # New player, assign new persistent ID
                    persistent_id = self.next_persistent_id
                    self.id_mapping[temp_id] = persistent_id
                    self.next_persistent_id += 1
                    logger.debug(f"New player detected: {temp_id} -> {persistent_id}")

            # Add to gallery
            self.add_to_gallery(persistent_id, embedding, frame_number, bbox, team)

            # Update detection with persistent ID
            detection['persistent_id'] = persistent_id
            detection['reid_confidence'] = similarity if matched_id else 1.0

            updated_detections.append(detection)

        return updated_detections

    def process_video(self, video_path: str, detections_path: str,
                     output_path: str) -> Dict:
        """
        Process entire video with re-identification.

        Args:
            video_path: Path to input video
            detections_path: Path to detection JSON
            output_path: Path to save re-identified detections

        Returns:
            Re-identified detection dictionary
        """
        # Load detections
        with open(detections_path, 'r') as f:
            detections = json.load(f)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        reid_detections = {}
        frame_idx = 0

        logger.info(f"Processing video with ReID: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for this frame
            frame_detections = detections.get(str(frame_idx), [])

            # Apply re-identification
            updated_detections = self.update_tracking(frame, frame_detections, frame_idx)

            reid_detections[frame_idx] = updated_detections

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames...")

        cap.release()

        # Save re-identified detections
        with open(output_path, 'w') as f:
            json.dump(reid_detections, f, indent=2)

        logger.info(f"ReID complete: {output_path}")
        logger.info(f"Total unique players identified: {len(self.player_gallery)}")

        return reid_detections

    def get_statistics(self) -> Dict:
        """Get ReID statistics."""
        stats = {
            'total_players': len(self.player_gallery),
            'total_embeddings': sum(len(embs) for embs in self.player_gallery.values()),
            'id_mappings': len(self.id_mapping)
        }

        # Per-player embedding counts
        player_counts = {
            pid: len(embs) for pid, embs in self.player_gallery.items()
        }
        stats['player_embedding_counts'] = player_counts

        return stats


def main():
    """Example usage of player ReID."""
    import argparse
    parser = argparse.ArgumentParser(description='Player Re-Identification with MobileNetV3')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--detections', required=True, help='Path to detections JSON')
    parser.add_argument('--output', default='outputs/reid_detections.json',
                       help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Similarity threshold')
    parser.add_argument('--no-mobilenet', action='store_true',
                       help='Disable MobileNetV3 (use manual features)')
    parser.add_argument('--no-faiss', action='store_true',
                       help='Disable Faiss (use sklearn)')
    args = parser.parse_args()

    # Run ReID
    reid = PlayerReID(
        similarity_threshold=args.threshold,
        use_mobilenet=not args.no_mobilenet,
        use_faiss=not args.no_faiss
    )
    reid.process_video(args.video, args.detections, args.output)

    # Print statistics
    stats = reid.get_statistics()
    print("\nReID Statistics:")
    print(f"Total unique players: {stats['total_players']}")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"ID mappings: {stats['id_mappings']}")


if __name__ == '__main__':
    main()
