import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

from pdf2image import convert_from_bytes

# srcディレクトリが見えるようにパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing.site_processor import SiteProcessor

def main():
    st.title("住宅デザインAIジェネレータ")
    st.write("土地図をアップロードすると、エッジ検出＆線分抽出を行います。")

    # A3比率(約1.414)を維持するため、(840, 594)に変更
    # sigma=1.0のまま
    processor = SiteProcessor(target_size=(1680, 1188), canny_sigma=1.0)

    uploaded_file = st.file_uploader("土地図をアップロード", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            # DPI=300で高解像度変換
            pages = convert_from_bytes(pdf_bytes, dpi=300)
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

        # 3) ノイズ除去 + モルフォロジー(Open→Close)
        cleaned_float = processor.remove_small_components(edges_float, min_area=5)
        cleaned_float = processor.morph_process(cleaned_float, kernel_size=3, op_type="open")
        cleaned_float = processor.morph_process(cleaned_float, kernel_size=3, op_type="close")
        cleaned_uint8 = (cleaned_float * 255).astype(np.uint8)

        # 4) Hough変換で直線抽出
        lines_img = processor.detect_lines(cleaned_float)

        with col3:
            st.subheader("直線抽出")
            st.image(lines_img, use_column_width=True)

        # 中間結果を表示
        st.write("---")
        st.write("### ノイズ除去 + モルフォロジー後")
        st.image(cleaned_uint8, caption="Open→Close処理後の2値画像", use_column_width=True)

if __name__ == "__main__":
    main()