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
        戻り値: float32の2次元配列(0.0～1.0)でエッジを表す
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # リサイズ
        resized = cv2.resize(gray, self.target_size)

        # 0～1に正規化してCannyエッジ
        resized_f = resized / 255.0
        edges = feature.canny(resized_f, sigma=2)

        # boolをfloat32にキャスト (0.0 or 1.0)
        edges_float = edges.astype(np.float32)
        return edges_float

    def remove_small_components(self, edges_float: np.ndarray, min_area=20) -> np.ndarray:
        """
        小さな連結成分を除去してノイズを減らす。
        edges_float: (0.0 or 1.0)の2次元配列
        min_area: 除去対象とする小さな成分の面積閾値
        戻り値: 小さい成分を除去した float32(0.0 or 1.0)配列
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)
        # 連結成分をラベリング
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            edges_uint8, connectivity=8
        )

        cleaned = np.zeros_like(edges_uint8)
        # ラベル0は背景、それ以外がオブジェクト
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 255

        # float32(0.0 or 1.0)に戻す
        cleaned_float = (cleaned > 0).astype(np.float32)
        return cleaned_float

    def morph_process(self, edges_float: np.ndarray, kernel_size=3, op_type="open") -> np.ndarray:
        """
        モルフォロジー処理(開閉)でノイズ除去や線の補正を行う。
        edges_float: (0.0 or 1.0)の2次元配列
        kernel_size: カーネルの大きさ
        op_type: "open" or "close"
        戻り値: モルフォロジー処理後のfloat32(0.0 or 1.0)
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if op_type == "open":
            # オープニング: 小さい白領域を除去
            morphed = cv2.morphologyEx(edges_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            # クロージング: 白領域の隙間を埋める
            morphed = cv2.morphologyEx(edges_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)

        morphed_float = (morphed > 0).astype(np.float32)
        return morphed_float

    def detect_lines(self, edges_float: np.ndarray) -> np.ndarray:
        """
        Hough変換を用いて直線を検出し、結果を画像として返す。
        edges_float: process() の戻り値 (0.0～1.0の2値)
        戻り値: 直線を描画した画像 (3chのuint8)
        """
        edges_uint8 = (edges_float * 255).astype(np.uint8)

        # Hough変換 (確率的HoughLinesP)
        # 文字が多い場合は thresholdを上げる、minLineLengthを大きくする、など調整
        lines = cv2.HoughLinesP(
            edges_uint8,
            rho=1,                # 1ピクセル刻み
            theta=np.pi / 180,    # 1度刻み
            threshold=120,        # 投票数
            minLineLength=50,     # 最小線分長
            maxLineGap=10         # 線分間の最大ギャップ
        )

        # 直線を描画するための画像 (3チャンネル黒画面) を用意
        h, w = edges_uint8.shape
        lines_img = np.zeros((h, w, 3), dtype=np.uint8)

        # 検出された各線分を白色で描画
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        return lines_img