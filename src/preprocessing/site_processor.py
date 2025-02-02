# src/preprocessing/site_processor.py
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

    def detect_lines(self, edges_float: np.ndarray) -> np.ndarray:
        """
        Hough変換を用いて直線を検出し、結果を画像として返す。
        edges_float: process() の戻り値 (0.0～1.0の2値)
        戻り値: 直線を描画した画像 (3chのuint8)
        """
        # float32(0.0/1.0) -> uint8(0/255) に変換
        edges_uint8 = (edges_float * 255).astype(np.uint8)

        # Hough変換 (確率的HoughLinesP)
        # パラメータは用途に合わせて調整してください
        lines = cv2.HoughLinesP(
            edges_uint8,
            rho=1,              # 1ピクセル刻み
            theta=np.pi / 180,  # 1度刻み
            threshold=80,       # 直線とみなす最低投票数
            minLineLength=30,   # 最小線分長
            maxLineGap=10       # 線分間の最大ギャップ
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