import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

from pdf2image import convert_from_bytes

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing.site_processor import SiteProcessor

def main():
    st.title("住宅デザインAIジェネレータ")
    st.write("土地図をアップロードすると、エッジ検出＆線分抽出を行います。")

    # SiteProcessorをsigma=1.0で初期化
    processor = SiteProcessor(target_size=(256, 256), canny_sigma=1.0)

    uploaded_file = st.file_uploader("土地図をアップロード", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            pages = convert_from_bytes(pdf_bytes, dpi=300)
            page = pages[0]
            image = page
            image_array = np.array(page)
        else:
            image = Image.open(uploaded_file)
            image_array = np.array(image)

        # 下部15%をカットして説明部分を除外
        h = image_array.shape[0]
        cutoff = int(h * 0.85)  # 上から85%の位置でカット
        # スライスして下15%を無視
        image_array = image_array[:cutoff, :]

        # オリジナルの(切り抜き後)画像表示用に再度PIL化
        cut_pil = Image.fromarray(image_array)

        # レイアウト用3カラム
        col1, col2, col3 = st.columns(3)

        # 1) オリジナル画像表示（下15%カット済み）
        with col1:
            st.subheader("土地図（下部カット）")
            st.image(cut_pil, use_column_width=True)

        # 2) エッジ抽出
        edges_float = processor.process(image_array)
        edges_uint8 = (edges_float * 255).astype(np.uint8)

        with col2:
            st.subheader("エッジ抽出")
            st.image(edges_uint8, use_column_width=True)

        # 3) ノイズ除去 (小領域除去) + Open→Close のモルフォロジー
        cleaned_float = processor.remove_small_components(edges_float, min_area=5)

        # まずOpenで小さい白領域を除去
        cleaned_float = processor.morph_process(cleaned_float, kernel_size=3, op_type="open")
        # 続けてCloseで線の切れ目を繋ぐ
        cleaned_float = processor.morph_process(cleaned_float, kernel_size=3, op_type="close")

        cleaned_uint8 = (cleaned_float * 255).astype(np.uint8)

        # 4) Hough変換で直線検出
        lines_img = processor.detect_lines(cleaned_float)

        with col3:
            st.subheader("直線抽出")
            st.image(lines_img, use_column_width=True)

        # 中間結果
        st.write("---")
        st.write("### ノイズ除去 + モルフォロジー後")
        st.image(cleaned_uint8, caption="Open→Close処理後の2値画像", use_column_width=True)

if __name__ == "__main__":
    main()