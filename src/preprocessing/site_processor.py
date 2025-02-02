import cv2
import numpy as np
from skimage import feature

class SiteProcessor:
    def __init__(self, target_size=(1024, 1024), canny_sigma=1.0):
        self.target_size = target_size
        # Cannyのsigmaを外部から指定できるように
        self.canny_sigma = canny_sigma

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Grayscale & Resize & Canny edge
        Return: float32 2D array (0.0 or 1.0)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # リサイズ
        resized = cv2.resize(gray, self.target_size)

        # 0～1に正規化してCannyエッジ
        resized_f = resized / 255.0

        # sigmaを小さくしてより繊細なエッジを検出
        edges = feature.canny(resized_f, sigma=self.canny_sigma)

        edges_float = edges.astype(np.float32)
        return edges_float

    def remove_small_components(self, edges_float: np.ndarray, min_area=5) -> np.ndarray:
        """
        小さい連結成分を除去してノイズを減らす (面積閾値を下げる)
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_uint8, connectivity=8)

        cleaned = np.zeros_like(edges_uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 255

        cleaned_float = (cleaned > 0).astype(np.float32)
        return cleaned_float

    def morph_process(self, edges_float: np.ndarray, kernel_size=3, op_type="open") -> np.ndarray:
        """
        モルフォロジーでノイズ除去や線の補正を行う。
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if op_type == "open":
            # オープニングで小さい白領域を除去
            morphed = cv2.morphologyEx(edges_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            # クロージングで隙間を埋める
            morphed = cv2.morphologyEx(edges_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)

        morphed_float = (morphed > 0).astype(np.float32)
        return morphed_float

    def detect_lines(self, edges_float: np.ndarray) -> np.ndarray:
        """
        Hough変換で直線を検出し、線を描画した画像を返す (3ch uint8)
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)

        # パラメータを緩和して感度を高める
        lines = cv2.HoughLinesP(
            edges_uint8,
            rho=1,
            theta=np.pi / 180,
            threshold=60,        # 120 -> 60に下げ
            minLineLength=30,    # 50 -> 30に短く
            maxLineGap=20        # 10 -> 20に広げ
        )

        h, w = edges_uint8.shape
        lines_img = np.zeros((h, w, 3), dtype=np.uint8)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        return lines_img