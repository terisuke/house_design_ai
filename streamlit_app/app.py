import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# srcディレクトリが見えるようにパスを追加（必要に応じて修正）
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.site_processor import SiteProcessor

def main():
    st.title("House Design AI Generator")
    st.write("Hello, Streamlit!")

    processor = SiteProcessor()

    # ファイルアップローダ
    uploaded_file = st.file_uploader("Upload a site plan", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # PIL形式で読み込み
        image = Image.open(uploaded_file)
        # NumPy配列に変換 (RGB想定)
        image_array = np.array(image)

        st.subheader("Original Site Plan")
        st.image(image, use_column_width=True)

        # 前処理
        processed_array = processor.process(image_array)

        # 前処理結果を表示する
        st.subheader("Processed Site Plan (Edges)")
        # ストリームリットで白黒画像を表示するには0～255に変換してuint8にしておくと見やすい
        display_img = (processed_array * 255).astype(np.uint8)
        st.image(display_img, use_column_width=True)

if __name__ == "__main__":
    main()