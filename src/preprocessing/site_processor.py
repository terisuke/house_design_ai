import cv2
import numpy as np
from skimage import feature

class SiteProcessor:
    def __init__(self, target_size=(1680, 1188), canny_sigma=1.0):
        """
        コンストラクタ
        Args:
            target_size: リサイズ先の画像サイズ (width, height) 
                        A3比率(1.414)を維持した例: (840, 594)
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

        # グレースケール化後にCLAHEを適用
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # シャープ化
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharp = cv2.convertScaleAbs(laplacian)

        # 指定サイズにリサイズ (A3比率: 1680x1188 など)
        resized = cv2.resize(sharp, self.target_size)

        # 0～1に正規化
        resized_f = resized / 255.0

        # Sobelフィルタを使用してエッジを検出
        sobelx = cv2.Sobel(resized_f, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(resized_f, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(sobelx**2 + sobely**2)

        # bool -> float32 (0.0/1.0)
        edges_float = edges.astype(np.float32)
        return edges_float

    def remove_small_components(self, edges_float: np.ndarray, min_area=1) -> np.ndarray:
        """
        小さい連結成分を除去してノイズ低減
        Args:
            edges_float: (0.0 or 1.0)の2値画像
            min_area: 除去する最小面積閾値 (1にして実質スキップ)
        Returns: 除去後の float32(0.0 or 1.0)
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_uint8, connectivity=8)

        cleaned = np.zeros_like(edges_uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                # 文字っぽい成分を除外する判定を追加
                x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
                if bw == 0 or bh == 0:
                    continue
                    
                aspect = max(bw, bh) / float(min(bw, bh))  # 縦横比(大きい方/小さい方)
                fill_ratio = area / (bw * bh)              # 塗りつぶし率

                # 線分っぽいもの(縦横比が2.0以上 & 密度が0.6以下)のみ残す
                if aspect >= 300.0 and fill_ratio <= 0.0001:
                    cleaned[labels == i] = 255

        cleaned_float = (cleaned > 0).astype(np.float32)
        return cleaned_float

    def morph_process(self, edges_float: np.ndarray, kernel_size=3, op_type="close") -> np.ndarray:
        """
        モルフォロジー演算 (close) で線の隙間を埋める
        Args:
            edges_float: (0.0 or 1.0)の2値画像
            kernel_size: カーネルの大きさ
            op_type: 'close' のみ使用
        Returns: 処理後の float32(0.0 or 1.0)
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # クロージングのみ実施して隙間を強めに埋める
        morphed = cv2.morphologyEx(edges_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)

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

        # パラメータを大幅に緩和して感度を最大限に高める
        lines = cv2.HoughLinesP(
            edges_uint8,
            rho=1,
            theta=np.pi / 180,
            threshold=5,       # 投票数を大幅に下げる
            minLineLength=5,   # 短い線分も検出
            maxLineGap=100     # 大きな隙間を許容
        )

        h, w = edges_uint8.shape
        lines_img = np.zeros((h, w, 3), dtype=np.uint8)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        return lines_img

    def remove_textlike_components(self, edges_float: np.ndarray, min_aspect_ratio: float = 1.5, max_fill_ratio: float = 0.7) -> np.ndarray:
        """
        文字っぽい成分を除去する。文字判定を弱める => 線とみなす範囲を広げる
        Args:
            edges_float: (0.0 or 1.0)の2値画像
            min_aspect_ratio: 除去する最小縦横比
            max_fill_ratio: 除去する最大塗りつぶし率
        Returns: 除去後の float32(0.0 or 1.0)
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_uint8, connectivity=8)

        cleaned = np.zeros_like(edges_uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            if bw == 0 or bh == 0:
                continue

            aspect = max(bw, bh) / float(min(bw, bh))  # 縦横比(大きい方/小さい方)
            fill_ratio = area / (bw * bh)              # 塗りつぶし率

            # 「線」とみなす条件を調整
            if aspect >= min_aspect_ratio and fill_ratio <= max_fill_ratio:
                cleaned[labels == i] = 255

        cleaned_float = (cleaned > 0).astype(np.float32)
        return cleaned_float