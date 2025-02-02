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
    st.title("House Design AI Generator")
    st.write("Hello, Streamlit!")

    processor = SiteProcessor()

    # ファイルアップローダ
    uploaded_file = st.file_uploader("Upload a site plan", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            # PDFファイルの場合
            pdf_bytes = uploaded_file.read()

            # pdf2imageでバイト列からページを画像に変換 (dpi=100で解像度指定、必要に応じて調整)
            pages = convert_from_bytes(pdf_bytes, dpi=100)
            
            # 今回は1ページ目のみ処理すると仮定
            page = pages[0]  # pageはPIL.Image形式
            
            # NumPy配列に変換
            image_array = np.array(page)
            # Streamlitで「オリジナル画像」として表示する際はPIL Imageを使えるので、変数名をimageにしておく
            image = page

        else:
            # PNG/JPGの場合
            image = Image.open(uploaded_file)
            image_array = np.array(image)

        # 元画像表示
        st.subheader("Original Site Plan")
        st.image(image, use_column_width=True)

        # 前処理
        processed_array = processor.process(image_array)

        # 前処理結果表示
        st.subheader("Processed Site Plan (Edges)")
        # ストリームリットで白黒画像を表示するには 0～255 の uint8 に変換しておくほうが見やすい
        display_img = (processed_array * 255).astype(np.uint8)
        st.image(display_img, use_column_width=True)

if __name__ == "__main__":
    main()