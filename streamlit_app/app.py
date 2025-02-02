import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# pdf2imageをインポート
from pdf2image import convert_from_bytes

# srcディレクトリが見えるようにパスを追加（必要に応じて修正）
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.site_processor import SiteProcessor

def main():
    st.title("住宅デザインAIジェネレータ")
    st.write("このアプリは、土地図をアップロードすると、その土地図をもとに住宅デザインを生成します。")
    st.write("今は土地図のエッジを抽出しています。")

    processor = SiteProcessor()

    # ファイルアップローダ
    uploaded_file = st.file_uploader(
        "土地図をアップロード", 
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type

        # --------------------------------------------------
        # (1) 画像読み込み
        # --------------------------------------------------
        if file_type == "application/pdf":
            # PDFファイルの場合
            pdf_bytes = uploaded_file.read()
            pages = convert_from_bytes(pdf_bytes, dpi=100)
            page = pages[0]  # 1ページ目のみ対象
            image = page
            image_array = np.array(page)
        else:
            # PNG/JPGの場合
            image = Image.open(uploaded_file)
            image_array = np.array(image)

        # --------------------------------------------------
        # (2) 画像表示（3列）
        # --------------------------------------------------
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("土地図")
            st.image(image, use_column_width=True)

        # --------------------------------------------------
        # (3) 前処理 (Cannyエッジ)
        # --------------------------------------------------
        processed_array = processor.process(image_array)
        # 表示用に 0～255 のuint8画像へ
        display_edges = (processed_array * 255).astype(np.uint8)

        with col2:
            st.subheader("エッジ抽出")
            st.image(display_edges, use_column_width=True)

        # --------------------------------------------------
        # (4) Hough変換で直線抽出 + 表示
        # --------------------------------------------------
        lines_img = processor.detect_lines(processed_array)
        
        with col3:
            st.subheader("直線抽出")
            # lines_imgは3chのuint8画像
            st.image(lines_img, use_column_width=True)

if __name__ == "__main__":
    main()