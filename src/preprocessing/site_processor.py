import cv2
import numpy as np
from skimage import feature

class SiteProcessor:
    def __init__(self, target_size=(1024, 1024), canny_sigma=1.0):
        """
        コンストラクタ
        Args:
            target_size: リサイズ先の画像サイズ (width, height)
            canny_sigma: Cannyエッジ検出のsigmaパラメータ
        """
        self.target_size = target_size
        self.canny_sigma = canny_sigma

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        画像をグレースケール→リサイズ→Cannyエッジ検出
        Returns: float32 2D array (0.0 or 1.0)
        """
        # グレースケール化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 指定サイズにリサイズ
        resized = cv2.resize(gray, self.target_size)

        # 0～1に正規化
        resized_f = resized / 255.0

        # Cannyエッジ (sigmaでエッジの繊細さを調整)
        edges = feature.canny(resized_f, sigma=self.canny_sigma)

        # bool -> float32 (0.0/1.0)
        edges_float = edges.astype(np.float32)
        return edges_float

    def remove_small_components(self, edges_float: np.ndarray, min_area=5) -> np.ndarray:
        """
        小さい連結成分を除去してノイズ低減
        Args:
            edges_float: (0.0 or 1.0)の2値画像
            min_area: 除去する最小面積閾値
        Returns: 除去後の float32(0.0 or 1.0)
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
        モルフォロジー演算 (open / close) でノイズ除去や線の補強
        Args:
            edges_float: (0.0 or 1.0)の2値画像
            kernel_size: カーネルの大きさ
            op_type: 'open' or 'close'
        Returns: 処理後の float32(0.0 or 1.0)
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if op_type == "open":
            # オープニング: 小さな白領域を消す
            morphed = cv2.morphologyEx(edges_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            # クロージング: 白領域の隙間を埋める
            morphed = cv2.morphologyEx(edges_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)

        morphed_float = (morphed > 0).astype(np.float32)
        return morphed_float

    def detect_lines(self, edges_float: np.ndarray) -> np.ndarray:
        """
        Hough変換(確率的)で直線検出し、画像に描画したものを返す
        Args:
            edges_float: (0.0 or 1.0)の2値画像
        Returns: 直線を描画したRGB画像 (uint8)
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)

        lines = cv2.HoughLinesP(
            edges_uint8,
            rho=1,
            theta=np.pi / 180,
            threshold=60,      # 投票数 (小さくするほど敏感)
            minLineLength=30,  # 検出される最小線分長
            maxLineGap=20      # 線分間の最大ギャップ
        )

        h, w = edges_uint8.shape
        lines_img = np.zeros((h, w, 3), dtype=np.uint8)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        return lines_img