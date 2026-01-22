import re
import time

import mlx.core as mx
import streamlit as st
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import argparse

title = "智能助手"
ver = "0.8.1"
debug = True


def generate(the_prompt, the_model):
    tokens = []
    skip = 0
    count = 0
    
    # 简化的重复检测：只检测连续的完全重复
    last_complete_response = ""
    repeat_count = 0
    max_repeats = 2
    
    # 编码提示
    input_ids = mx.array(tokenizer.encode(the_prompt))
    
    # 创建采样器，设置用户要求的参数
    # 参数说明：
    # - temp: 温度参数，控制生成的随机性，0.6 是一个平衡值
    # - top_p: 核采样参数，控制生成的多样性，0.95 保留 95% 的概率质量
    # - top_k: 保留概率最高的 k 个 token，20 是一个适中值
    # - min_p: 最小概率阈值，0 表示不使用此功能
    # - min_tokens_to_keep: 最小保留的 token 数，使用默认值 1
    sampler = make_sampler(
        temp=0.6,
        top_p=0.95,
        min_p=0,
        min_tokens_to_keep=1,
        top_k=20,
        xtc_probability=0.0,
        xtc_threshold=0.0,
        xtc_special_tokens=[]
    )
    
    # 开始生成，传递采样器
    gen = generate_step(
        input_ids, 
        the_model, 
        sampler=sampler,
        max_tokens=context_length  # 设置最大生成的 token 数
    )
    
    # 循环生成，直到生成器停止或达到最大 token 数
    for token, prob in gen:
        tokens.append(token)
        text = tokenizer.decode(tokens)
        current_chunk = text[skip:]
        
        # 输出当前生成的文本
        yield current_chunk
        
        # 更新偏移量和计数
        skip = len(text)
        count += 1
        
        # 检查是否达到最大 token 数
        if count >= context_length:
            break


def show_chat(the_prompt, previous=""):
    if debug:
        print(the_prompt)
        print("-" * 80)

    with ((st.chat_message("assistant"))):
        message_placeholder = st.empty()
        response = previous

        # 生成并显示内容
        for chunk in generate(the_prompt, model):
            response = response + chunk

            if not previous:
                # begin neural-beagle-14 fixes
                response = re.sub(r"^/\*+/", "", response)
                response = re.sub(r"^:+", "", response)
                # end neural-beagle-14 fixes

            # 移除所有不需要的标签
            response = re.sub(r"<think>", "", response)
            response = re.sub(r"</think>", "", response)
            response = re.sub(r"<\|im_start\|>", "", response)
            response = re.sub(r"<\|im_end\|>", "", response)
            response = re.sub(r"<\|endoftext\|>", "", response)
            response = re.sub(r"<s>", "", response)
            response = re.sub(r"</s>", "", response)
            
            # 移除重复的 "Human:" 标记
            response = re.sub(r"Human:", "", response)
            
            # 移除多余的空行
            response = re.sub(r"\n{3,}", "\n\n", response)
            
            # 移除特殊字符
            response = response.replace('�', '')
            
            # 实时显示生成的内容
            message_placeholder.markdown(response + "▌")

        # 生成完成后，清理最终内容
        # 1. 移除多余的空行
        final_response = re.sub(r"\n{3,}", "\n\n", response)
        
        # 2. 确保内容格式正确
        final_response = final_response.strip()
        
        # 3. 显示最终清理后的内容
        message_placeholder.markdown(final_response)

    # 将最终内容添加到会话状态
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    
    # 移除自动继续生成逻辑，改为通过调整生成参数来避免中途停止
    # 这样可以避免重复内容问题


def remove_last_occurrence(array, criteria_fn):
    for i in reversed(range(len(array))):
        if criteria_fn(array[i]):
            del array[i]
            break


def build_memory():
    # 限制对话历史的长度，只保留最近的 5 条消息
    max_history_length = 5
    if len(st.session_state.messages) > 2:
        # 保留最近的 max_history_length 条消息
        return st.session_state.messages[max(1, len(st.session_state.messages) - max_history_length):-1]
    return []


def queue_chat(the_prompt, continuation=""):
    # workaround because the chat boxes are not really replaced until a rerun
    st.session_state["prompt"] = the_prompt
    st.session_state["continuation"] = continuation
    st.rerun()


# tx @cocktailpeanut
parser = argparse.ArgumentParser(description="mlx-ui")
parser.add_argument("--models", type=str, help="the txt file that contains the models list", default="models.txt")
args = parser.parse_args()
models_file = args.models

assistant_greeting = "我能为您提供什么帮助？"

with open(models_file, 'r') as file:
    model_refs = [line.strip() for line in file.readlines() if not line.startswith('#')]

model_refs = {k.strip(): v.strip() for k, v in [line.split("|") for line in model_refs]}

st.set_page_config(
    page_title=title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(title)

st.markdown("""
<style>
.stDeployButton{display:none}
/* 修改运行状态文本为中文 */
[data-testid='stStatusWidget'] {
    position: relative;
}
[data-testid='stStatusWidget'] span,
[data-testid='stStatusWidget'] div {
    display: none !important;
}
[data-testid='stStatusWidget']::before {
    content: '运行中...';
    display: inline-block;
    margin-right: 10px;
}
[data-testid='stStatusWidget'] button {
    font-size: 14px !important;
}
[data-testid='stStatusWidget'] button span {
    display: none !important;
}
[data-testid='stStatusWidget'] button::after {
    content: '停止';
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


import os

@st.cache_resource(show_spinner=True)
def load_model_and_cache(ref):
    # 展开本地路径中的 ~ 符号
    if os.path.exists(os.path.expanduser(ref)):
        ref = os.path.expanduser(ref)
    return load(ref, {"trust_remote_code": True})


model = None

model_ref = st.sidebar.selectbox("模型", model_refs.keys(), format_func=lambda value: model_refs[value],
                                 help="查看 https://modelscope.cn 获取更多模型。将您喜欢的模型添加到 models.txt 文件中。")

if model_ref.strip() != "-":
    model, tokenizer = load_model_and_cache(model_ref)

    chat_template = tokenizer.chat_template or (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )
    supports_system_role = "system role not supported" not in chat_template.lower()

    system_prompt = st.sidebar.text_area("系统提示", "你是一位智慧的AI助手，基于大量人类知识训练而成。在回答问题时，请直接给出最终答案，不需要使用任何特殊标签或标记。回答要简洁明了，直接针对问题。重要：不要重复之前的内容，不要重复相同的段落或句子。",
                                         disabled=not supports_system_role)

    context_length = st.sidebar.number_input('上下文长度', value=2048, min_value=99, step=100, max_value=32000,
                                             help="大致打印的最大单词数。")

    st.sidebar.markdown("---")
    actions = st.sidebar.columns(2)

    # give a bit of time for sidebar widgets to render
    time.sleep(0.05)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": assistant_greeting}]

    stop_words = ["<|im_start|>", "<|im_end|>", "<s>", "</s>"]

    if actions[0].button("😶‍🌫️ 清空", use_container_width=True,
                         help="清空之前的对话。"):
        st.session_state.messages = [{"role": "assistant", "content": assistant_greeting}]
        if "prompt" in st.session_state and st.session_state["prompt"]:
            st.session_state["prompt"] = None
            st.session_state["continuation"] = None
        st.rerun()

    if actions[1].button("🔂 继续", use_container_width=True,
                         help="继续生成。"):

        user_prompts = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]

        if user_prompts:

            last_user_prompt = user_prompts[-1]

            assistant_responses = [msg["content"] for msg in st.session_state.messages
                                   if msg["role"] == "assistant" and msg["content"] != assistant_greeting]
            last_assistant_response = assistant_responses[-1] if assistant_responses else ""

            # remove last line completely, so it is regenerated correctly (in case it stopped mid-word or mid-number)
            last_assistant_response_lines = last_assistant_response.split('\n')
            if len(last_assistant_response_lines) > 1:
                last_assistant_response_lines.pop()
                last_assistant_response = "\n".join(last_assistant_response_lines)

            messages = [
                {"role": "user", "content": last_user_prompt},
                {"role": "assistant", "content": last_assistant_response},
            ]
            if supports_system_role:
                messages.insert(0, {"role": "system", "content": system_prompt})

            full_prompt = tokenizer.apply_chat_template(messages,
                                                        tokenize=False,
                                                        add_generation_prompt=False,
                                                        chat_template=chat_template)
            full_prompt = full_prompt.rstrip("\n")

            # remove last assistant response from state, as it will be replaced with a continued one
            remove_last_occurrence(st.session_state.messages,
                                   lambda msg: msg["role"] == "assistant" and msg["content"] != assistant_greeting)

            queue_chat(full_prompt, last_assistant_response)

    if prompt := st.chat_input("聊点什么..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        messages = []
        if supports_system_role:
            messages += [{"role": "system", "content": system_prompt}]
        messages += build_memory()
        messages += [{"role": "user", "content": prompt}]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    chat_template=chat_template)
        full_prompt = full_prompt.rstrip("\n")

        queue_chat(full_prompt)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # give a bit of time for messages to render
    time.sleep(0.05)

    if "prompt" in st.session_state and st.session_state["prompt"]:
        show_chat(st.session_state["prompt"], st.session_state["continuation"])
        st.session_state["prompt"] = None
        st.session_state["continuation"] = None

st.sidebar.markdown("---")
st.sidebar.markdown(f"版本 v{ver} / Streamlit {st.__version__}")
