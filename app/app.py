import gradio as gr
import ollama
import base64
import io

# 設定模型
MODEL_VARIANT = "gemma4:e4b" 

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def solve_math_problem(image, language, difficulty):
    if image is None:
        yield "⚠️ 請先上傳圖片。"
        return

    try:
        base64_image = image_to_base64(image)

        # 嚴格限制格式的 Prompt
        prompt = f"""
        你是一位精確的數學老師。
        
        【開場白】
        "Hello! 📐 我是您的數學小老師。看到這道題目，您可能覺得有點複雜，但其實它是一個非常經典的問題。不用擔心，我會一步一步地為您解釋，不僅教您「怎麼做」，更會告訴您「為什麼這樣做」，讓您學會掌握這個概念！"

        【回答規範】
        1. 🔎 **題目要求**：簡短說明題目問什麼。
        2. 💡 **使用公式**：只列出本題需要用到的數學公式 (使用 LaTeX $$ 格式)。
        3. ✍️ **逐步詳解**：清晰的計算步驟。
        4. ✅ **最終答案**：使用 \\boxed{{}} 包裹答案。

        【數學格式注意事項】
        - 必須使用 LaTeX 格式。
        - 行內符號用 $ $ (例如 $x=5$)。
        - 獨立公式塊用 $$ $$ (例如 $$x = \\frac{{-b \\pm \\sqrt{{b^2-4ac}}}}{{2a}}$$)。
        - 不要使用任何特殊的 Markdown 列表符號在公式內部。
        
        回答語言：{language}，程度：{difficulty}。
        """

        response = ollama.chat(
            model=MODEL_VARIANT,
            messages=[{'role': 'user', 'content': prompt, 'images': [base64_image]}],
            stream=True
        )

        full_response = ""
        for chunk in response:
            content = chunk['message']['content']
            # 簡單的 LaTeX 修正邏輯，確保顯示正確
            content = content.replace(r"$$", "$").replace(r"$$", "$")
            content = content.replace(r"$$", "$$").replace(r"$$", "$$")
            
            full_response += content
            yield full_response

    except Exception as e:
        yield f"❌ 發生錯誤：{str(e)}"

# ==================== 優化後的介面 ====================

# 修正閃爍問題的 CSS
custom_css = """
.stable-markdown {
    min-height: 600px !imortant; 
    overflow-y: auto !important;
    display: block !important;
    line-height: 1.6;
}
/* 防止內容更新時容器塌陷 */
.stable-markdown div {
    min-height: 100%;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📐 Gemma 4 專業數學解題v0")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="拍照或上傳題目")
            with gr.Row():
                lang_opt = gr.Dropdown(["繁體中文", "English"], value="繁體中文", label="語言")
                diff_opt = gr.Dropdown(["國中", "高中"], value="國中", label="程度")
            submit_btn = gr.Button("🔍 啟動 Gemma 4 分析", variant="primary")
            
            gr.Info("提示：本地運算需要時間，請觀察上方進度條。")
        
        with gr.Column(scale=2):
            # 這裡加入了關鍵的 latex_delimiters
            output_md = gr.Markdown(
                label="AI 老師解答區",
                elem_classes="stable-markdown",
                latex_delimiters=[
                    {"left": "$", "right": "$", "display": False},
                    {"left": "$$", "right": "$$", "display": True}
                ]
            )

    submit_btn.click(
        fn=solve_math_problem,
        inputs=[input_image, lang_opt, diff_opt],
        outputs=[output_md],
        scroll_to_output=False # 禁止滾動以減少閃爍
    )

if __name__ == "__main__":
    demo.launch()
