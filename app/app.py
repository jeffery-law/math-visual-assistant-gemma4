import gradio as gr
import ollama
import base64
import io
import time
import csv
import os
from datetime import datetime

# ==================== 路徑與設定 ====================
MODEL_VARIANT = "gemma4:e4b" 
BASE_LOG_PATH = r"C:\Users\user\Desktop\project\math-visual-assistant-gemma4\outputs\log"
PERF_LOG_FILE = os.path.join(BASE_LOG_PATH, "performance_log.csv")
OUT_LOG_FILE = os.path.join(BASE_LOG_PATH, "output_log.csv")

os.makedirs(BASE_LOG_PATH, exist_ok=True)

def init_csv():
    if not os.path.exists(PERF_LOG_FILE):
        with open(PERF_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Language", "Style", "Start_Time", "End_Time", "Duration_Sec"])
    if not os.path.exists(OUT_LOG_FILE):
        with open(OUT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "AI_Generated_Output"])

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def solve_math_problem(image, language, user_style):
    if image is None:
        yield "⚠️ Please upload an image."
        return

    init_csv()
    start_time_stamp = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    labels = {
        "繁體中文": ["🔎 題目要求", "💡 核心公式", "✍️ 逐步詳解", "✅ 最終答案"],
        "English": ["🔎 Task Requirement", "💡 Core Formula", "✍️ Step-by-Step Solution", "✅ Final Answer"]
    }
    cur = labels.get(language, labels["English"])

    try:
        base64_image = image_to_base64(image)

        # Prompt instruction to avoid breaking LaTeX mid-stream
        prompt = f"""
        Explain this math problem. 
        Style: {user_style}
        Language: {language}

        Rules:
        1. Use $...$ for inline math.
        2. Use $$...$$ for block equations.
        3. Do not add extra spaces inside LaTeX symbols.
        
        Structure:
        ---
        ## {cur[0]}
        ---
        ## {cur[1]}
        ---
        ## {cur[2]}
        ---
        ## {cur[3]}
        """

        response = ollama.chat(
            model=MODEL_VARIANT,
            messages=[{'role': 'user', 'content': prompt, 'images': [base64_image]}],
            stream=True
        )

        full_response = ""
        for chunk in response:
            if 'message' in chunk:
                full_response += chunk['message']['content']
                yield full_response

        # Logging
        duration = round(time.time() - start_time_stamp, 2)
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(PERF_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([now_ts, language, user_style, start_time_str, now_ts, duration])
        
        with open(OUT_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f).writerow([now_ts, full_response])

    except Exception as e:
        yield f"❌ Error: {str(e)}"

# ==================== UI 介面設定 ====================

custom_css = """
#fixed-box {
    height: 600px !important;
    overflow-y: auto !important;
    border: 2px solid #333 !important;
    padding: 20px !important;
    background: #ffffff !important;
    /* Changed from pre-wrap to normal to fix shaking and vertical stacking */
    white-space: normal !important; 
}
#fixed-box p {
    margin-bottom: 8px !important;
    line-height: 1.6 !important;
}
hr {
    border: 0;
    height: 2px;
    background: #444;
    margin: 10px 0;
}
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📐 Gemma 4 Math Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Capture Math Problem")
            lang_opt = gr.Radio(["繁體中文", "English"], value="繁體中文", label="Language")
            style_opt = gr.Dropdown(
                choices=["Beginner", "Intermediate", "Expert"], 
                value="Intermediate", 
                label="Style"
            )
            submit_btn = gr.Button("🔍 Solve", variant="primary")
        
        with gr.Column(scale=2):
            output_md = gr.Markdown(
                elem_id="fixed-box",
                latex_delimiters=[
                    {"left": "$", "right": "$", "display": False},
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": r"$$", "right": r"$$", "display": False},
                    {"left": r"$$", "right": r"$$", "display": True}
                ]
            )

    submit_btn.click(
        fn=solve_math_problem,
        inputs=[input_image, lang_opt, style_opt],
        outputs=[output_md]
    )

if __name__ == "__main__":
    # Moved CSS to launch() for Gradio 6.0 compatibility
    demo.launch(server_port=7860, css=custom_css)
