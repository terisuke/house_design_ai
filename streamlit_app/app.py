import streamlit as st
import numpy as np
from PIL import Image
import os
import sys
import cv2

from pdf2image import convert_from_bytes

# srcディレクトリが見えるようにパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing.site_processor import SiteProcessor

def main():
    st.title("住宅デザインAIジェネレータ")
    st.write("土地図をアップロードすると、エッジ検出＆線分抽出を行います。")

    # target_size=(1680, 1188), canny_sigma=1.0 はそのまま
    processor = SiteProcessor(target_size=(1680, 1188), canny_sigma=1.0)

    uploaded_file = st.file_uploader("土地図をアップロード", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            pages = convert_from_bytes(pdf_bytes, dpi=1000)
            page = pages[0]
            image = page
            image_array = np.array(page)
        else:
            image = Image.open(uploaded_file)
            image_array = np.array(image)

        # レイアウト: 3カラム
        col1, col2, col3 = st.columns(3)

        # 1) オリジナル画像
        with col1:
            st.subheader("土地図")
            st.image(image, use_column_width=True)

        # 2) エッジ抽出
        edges_float = processor.process(image_array)
        edges_uint8 = (edges_float * 255).astype(np.uint8)
        with col2:
            st.subheader("エッジ抽出")
            st.image(edges_uint8, use_column_width=True)

        # 3) ノイズ除去 (min_area=1) + Morph(Close only)
        cleaned_float = processor.remove_small_components(edges_float, min_area=1)
        cleaned_float = processor.morph_process(cleaned_float, kernel_size=7, op_type="close")
        cleaned_float = processor.morph_process(cleaned_float, kernel_size=7, op_type="close")
        # 文字っぽい成分除去 (閾値を緩和)
        # cleaned_float = processor.remove_textlike_components(cleaned_float, min_aspect_ratio=300.0, max_fill_ratio=0.0001)
        cleaned_uint8 = (cleaned_float * 255).astype(np.uint8)

        # 4) Hough変換で直線抽出
        lines = cv2.HoughLinesP(
            edges_uint8,
            rho=0.5,            # 分解能を細かく
            theta=np.pi / 360,  # 分解能を細かく
            threshold=1,        # 投票数を限界まで下げる
            minLineLength=1,    # 最短の線分も検出
            maxLineGap=200      # 非常に大きな隙間を許容
        )
        lines_img = processor.detect_lines(cleaned_float)

        with col3:
            st.subheader("直線抽出")
            st.image(lines_img, use_column_width=True)

        # 中間結果を表示
        st.write("---")
        st.write("### ノイズ除去 + モルフォロジー後")
        st.image(cleaned_uint8, caption="Closeのみ (iterations=2)", use_column_width=True)

if __name__ == "__main__":
    main()