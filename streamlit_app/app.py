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
    st.write("このアプリは、土地図をアップロードすると、その土地図をもとに住宅デザインを生成します。")
    st.write("今は土地図のエッジを抽出＆直線検出（Hough）を行い、更にノイズを減らしてみます。")

    processor = SiteProcessor()

    # ファイルアップローダ
    uploaded_file = st.file_uploader(
        "土地図をアップロード", 
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            pages = convert_from_bytes(pdf_bytes, dpi=100)
            page = pages[0]
            image = page
            image_array = np.array(page)
        else:
            image = Image.open(uploaded_file)
            image_array = np.array(image)

        # --- レイアウト用カラム ---
        col1, col2, col3 = st.columns(3)

        # 1) オリジナル画像表示
        with col1:
            st.subheader("土地図")
            st.image(image, use_column_width=True)

        # 2) Cannyエッジ抽出
        edges_float = processor.process(image_array)
        edges_uint8 = (edges_float * 255).astype(np.uint8)

        with col2:
            st.subheader("エッジ抽出")
            st.image(edges_uint8, use_column_width=True)

        # 3) 小領域除去 + モルフォロジー演算(オープニング)でノイズ軽減
        cleaned_float = processor.remove_small_components(edges_float, min_area=10)
        cleaned_float = processor.morph_process(cleaned_float, kernel_size=3, op_type="open")

        cleaned_uint8 = (cleaned_float * 255).astype(np.uint8)

        # 4) 直線検出 (Hough変換)
        lines_img = processor.detect_lines(cleaned_float)

        with col3:
            st.subheader("直線抽出")
            st.image(lines_img, use_column_width=True)

        # --- 追加でノイズ除去の中間結果などを見たい場合 ---
        st.write("---")
        st.write("### ノイズ除去後の2値画像")
        st.image(cleaned_uint8, caption="Remove small comps + Morphology(Open)", use_column_width=True)

if __name__ == "__main__":
    main()