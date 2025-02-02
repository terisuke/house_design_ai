import cv2
import numpy as np
from skimage import feature

class SiteProcessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process site plan image for model input.
        (例) グレースケール & リサイズ & エッジ検出
        """
        # もしRGB画像ならグレースケールへ
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # リサイズ
        resized = cv2.resize(gray, self.target_size)

        # スケールを0～1にしてからCannyでエッジ検出
        resized_f = resized / 255.0
        edges = feature.canny(resized_f, sigma=2)

        # boolをfloat32にキャスト
        edges_float = edges.astype(np.float32)

        return edges_float