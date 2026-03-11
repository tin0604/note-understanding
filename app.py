import streamlit as st
import base64
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
import io

# ---------- 页面配置 ----------
st.set_page_config(page_title="手写笔记理解助手", layout="centered")
st.title("📝 手写笔记理解辅助系统")
st.markdown("上传你的手写笔记照片，AI 将帮你整理知识点、题型和范围。")

# ---------- 从 secrets 读取 API 配置 ----------
API_KEY = st.secrets["BAIDU_API_KEY"]
BASE_URL = st.secrets.get("BAIDU_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3")
MODEL_NAME = "ernie-4.5-vl-28b-a3b-thinking"

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------- 图像预处理函数（可选） ----------
def preprocess_image(image: Image.Image) -> Image.Image:
    """使用 OpenCV 进行去噪、对比度增强（可根据需要开启）"""
    # 转换为 OpenCV 格式
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 自适应直方图均衡化（增强对比度）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 中值滤波去噪
    denoised = cv2.medianBlur(enhanced, 3)

    # 转回 RGB
    result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result)

# ---------- 图片编码函数 ----------
def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ---------- UI 布局 ----------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("选择笔记图片", type=["jpg", "jpeg", "png"])
    use_preprocess = st.checkbox("启用图像预处理（增强识别）", value=False)

with col2:
    prompt_text = st.text_area(
        "输入你的需求（可自定义）",
        value="这是我课上做的关于“期末考试内容”的笔记，由于是写在草稿本上的，可能会有些草稿的干扰，请帮我整理一下期末考试的题型和范围",
        height=150
    )

# 处理按钮
if st.button("🚀 开始理解笔记", type="primary"):
    if uploaded_file is None:
        st.warning("请先上传一张图片")
        st.stop()

    # 显示原始图片
    st.image(uploaded_file, caption="已上传图片", use_container_width=True)

    # 读取并预处理
    image = Image.open(uploaded_file)
    if use_preprocess:
        with st.spinner("正在进行图像预处理..."):
            image = preprocess_image(image)
        st.image(image, caption="预处理后图片", use_container_width=True)

    # 转换为 Base64
    img_base64 = encode_image_to_base64(image)

    # 调用 ERNIE 模型
    with st.spinner("AI 正在思考中，请稍候..."):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ],
                stream=False
            )

            response = completion.choices[0].message
            reasoning = getattr(response, 'reasoning_content', None)
            content = response.content

            # 显示结果
            st.success("理解完成！")

            if reasoning:
                with st.expander("🧠 AI 思考过程", expanded=False):
                    st.markdown(reasoning)

            st.subheader("📄 整理结果")
            st.markdown(content)

            # 提供下载
            result_text = f"【思考过程】\n{reasoning if reasoning else '无'}\n\n【整理结果】\n{content}"
            st.download_button(
                label="📥 下载结果为文本文件",
                data=result_text,
                file_name="笔记整理结果.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"调用失败: {e}")

# ---------- 页脚 ----------
st.markdown("---")
st.markdown("Powered by 百度文心 ERNIE-4.5-VL & Streamlit")