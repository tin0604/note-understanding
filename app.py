import streamlit as st
import base64
from PIL import Image
import openai
import io

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="手写笔记理解助手", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("📝 手写笔记理解辅助系统")
st.markdown("""
上传你的手写笔记照片，AI 将帮你整理知识点、题型和范围。
支持手机拍照上传或电脑文件上传，无需 VPN，随时随地可用。
""")

# ---------- 从 secrets 读取 API 配置 ----------
try:
    API_KEY = st.secrets["BAIDU_API_KEY"]
    BASE_URL = st.secrets.get("BAIDU_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3")
except Exception as e:
    st.error("⚠️ API 密钥未配置，请在 Streamlit Cloud 的 Secrets 中设置 BAIDU_API_KEY")
    st.stop()

MODEL_NAME = "ernie-4.5-vl-28b-a3b-thinking"

# ---------- 初始化 OpenAI 客户端 ----------
try:
    openai.api_key = API_KEY
    openai.api_base = BASE_URL
    st.success("✅ API 配置成功！")
except Exception as e:
    st.error(f"❌ API 配置失败: {str(e)}")
    st.stop()

# ---------- 图片编码函数 ----------
def encode_image_to_base64(image: Image.Image) -> str:
    """将 PIL Image 转换为 base64 字符串"""
    buffered = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ---------- 调用 ERNIE 的函数 ----------
def call_ernie_model(image_base64, prompt):
    """调用 ERNIE 模型"""
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            timeout=60
        )
        return response
    except Exception as e:
        st.error(f"API 调用失败: {str(e)}")
        return None

# ---------- UI 布局 ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传图片")
    uploaded_file = st.file_uploader(
        "选择笔记图片", 
        type=["jpg", "jpeg", "png"],
        help="支持手机拍照或电脑上传"
    )
    
    # 修复：将 use_container_width 改为 use_column_width
    if uploaded_file is not None:
        st.image(uploaded_file, caption="已上传图片", use_column_width=True)

with col2:
    st.subheader("✏️ 你的需求")
    
    template_options = [
        "自定义输入",
        "整理期末考试题型和范围",
        "提取笔记中的重点知识点",
        "将手写笔记转为结构化大纲",
        "总结这节课的核心内容"
    ]
    
    selected_template = st.selectbox("选择模板", template_options)
    
    if selected_template == "自定义输入":
        prompt_text = st.text_area(
            "输入你的具体需求",
            value="这是我课上做的笔记，请帮我整理主要内容、重点和难点",
            height=150
        )
    else:
        template_map = {
            "整理期末考试题型和范围": "这是我课上做的关于'期末考试内容'的笔记，由于是写在草稿本上的，可能会有些草稿的干扰，请帮我整理一下期末考试的题型和范围，用清晰的格式列出",
            "提取笔记中的重点知识点": "请提取这张笔记图片中的重点知识点，用要点形式列出，并标注重要性等级",
            "将手写笔记转为结构化大纲": "请将这张手写笔记转换为结构化的大纲形式，用标题和子标题组织内容",
            "总结这节课的核心内容": "请总结这张笔记图片中的核心内容，用简洁的语言概括主要观点"
        }
        prompt_text = template_map[selected_template]
        st.text_area("输入的需求", value=prompt_text, height=150, disabled=True)

# ---------- 处理按钮 ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    process_button = st.button("🚀 开始理解笔记", type="primary", use_container_width=True)

# ---------- 处理逻辑 ----------
if process_button:
    if uploaded_file is None:
        st.warning("请先上传一张图片")
        st.stop()
    
    with st.status("🔄 处理中...", expanded=True) as status:
        st.write("1. 读取图片...")
        image = Image.open(uploaded_file)
        
        st.write("2. 准备发送给 AI...")
        img_base64 = encode_image_to_base64(image)
        
        st.write("3. 调用百度文心 ERNIE 模型...")
        response = call_ernie_model(img_base64, prompt_text)
        
        if response is None:
            status.update(label="❌ 处理失败", state="error")
            st.stop()
        
        status.update(label="✅ 处理完成!", state="complete")
    
    # 解析响应
    try:
        if hasattr(response, 'choices'):
            message = response.choices[0].message
            reasoning = getattr(message, 'reasoning_content', None)
            content = message.get('content', '') if hasattr(message, 'get') else message.content
        else:
            reasoning = response.get('reasoning_content')
            content = response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"解析响应失败: {str(e)}")
        st.stop()
    
    # 显示结果
    st.success("✨ 理解完成！")
    
    result_col1, result_col2 = st.columns([1, 1])
    
    with result_col1:
        if reasoning:
            with st.expander("🧠 AI 思考过程", expanded=False):
                st.markdown(str(reasoning))
    
    with result_col2:
        if content:
            result_text = f"【思考过程】\n{reasoning if reasoning else '无'}\n\n【整理结果】\n{content}"
            st.download_button(
                label="📥 下载结果",
                data=result_text.encode('utf-8'),
                file_name="笔记整理结果.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    st.subheader("📄 整理结果")
    
    # 修复：确保 content 不为空
    if content:
        if content.strip().startswith('#') or '**' in content or '- ' in content or '1.' in content:
            st.markdown(content)
        else:
            st.text_area("", value=content, height=300, disabled=True)
    else:
        st.warning("无结果返回")

# ---------- 页脚 ----------
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by 百度文心 ERNIE-4.5-VL | 手写笔记理解辅助系统</p>
    <p style='font-size: 0.8em;'>支持手机拍照上传，随时随地整理笔记</p>
</div>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("📋 使用说明")
    st.markdown("""
    1. **上传图片**：点击上传按钮选择手写笔记照片
    2. **选择需求**：从模板中选择或自定义输入
    3. **开始理解**：点击按钮等待 AI 处理
    4. **查看结果**：AI 会展示思考过程和整理结果
    5. **下载保存**：可将结果保存为文本文件
    
    **支持格式**：JPG、JPEG、PNG
    """)
