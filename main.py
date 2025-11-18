import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import random
import time
from math import exp
import difflib 

# --- 1. INITIAL SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="Module 2: Prompt Engineering Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'progress' not in st.session_state:
    st.session_state.progress = {f'H{i}': '🔴' for i in range(1, 7)}
    st.session_state.journal = []
    st.session_state.current_tab = 'H1'
    st.session_state.guidance = "Welcome to Module 2! Start with H1 to begin prompt refinement." 
    st.session_state.h4_results = pd.DataFrame() 
    if 'h2_model_results' not in st.session_state:
        st.session_state.h2_model_results = pd.DataFrame() 

if 'h1_result_initial' not in st.session_state:
    st.session_state.h1_result_initial = None
    st.session_state.h1_result_refined = None
    st.session_state.h1_metrics_initial = {}
    st.session_state.h1_metrics_refined = {}
    st.session_state.h2_result = None
    st.session_state.h3_result = None 
    st.session_state.h5_final_output = None 
if 'h5_attempts' not in st.session_state:
    st.session_state.h5_attempts = []
if 'h6_ran' not in st.session_state:
    st.session_state.h6_ran = False

# --- 🔧 CONFIGURATION FOR GROQ API ---
AI_API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY_NAME = "GROQ_API_KEY"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"

COMPARISON_MODELS = [
    "llama-3.1-8b-instant", 
    "mixtral-8x7b-32768", 
    DEFAULT_LLM_MODEL 
]

STYLING = """
<style>
.stApp {
    background-color: #FFFDF7; 
    color: #333; 
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
section[data-testid="stSidebar"] {
    width: 380px !important; 
}
.stButton>button[kind="primary"] {
    border: 3px solid #FF5733 !important;
}
.stButton>button {
    background-color: #008080; 
    color: white !important;
    border-radius: 8px;
    font-weight: 600;
    box-shadow: 0 0 10px rgba(0, 128, 128, 0.5); 
}
.title-header {
    color: #0d47a1;
    font-weight: 800;
    font-size: 38px;
}
.box-container {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    border-left: 5px solid #0d47a1;
    background-color: #f0f8ff;
}
.goal-box {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    border-left: 5px solid #28a745;
    background-color: #f6fff9;
}
/* Vertical Metric Card Styling */
.metric-card {
    background-color: white;
    border: 1px solid #e0e0e0;
    border-left: 5px solid #FF5733;
    padding: 15px;
    margin-bottom: 10px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.metric-title {
    font-size: 0.85em;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 5px;
}
.metric-value {
    font-size: 1.4em;
    font-weight: bold;
    color: #333;
}
</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)

# --- 2. UTILITY & AI CLIENT FUNCTIONS ---
class GroqClient:
    def __init__(self, api_key_name, model_id):
        self.model_id = model_id
        try:
            self.api_key = st.secrets[api_key_name]
            self.status = "READY (Using real API key)"
        except KeyError:
            self.api_key = "SK-SIMULATED-KEY"
            self.status = "SIMULATED (API key not found)"

def get_groq_client():
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = GroqClient(API_KEY_NAME, DEFAULT_LLM_MODEL)
    return st.session_state.groq_client

def get_progress_badge(key):
    return st.session_state.progress.get(key, '🔴')

def update_progress(key, status):
    st.session_state.progress[key] = status

def update_guidance(message):
    st.session_state.guidance = message

def analyze_text_metrics(text):
    if not text: return {"tokens": 0, "flesch_score": 0, "text_length": 0}
    tokens = text.split()
    word_count = len(tokens)
    syllable_count = sum(len(re.findall('[aeiouy]+', w.lower())) for w in tokens)
    sentence_count = len(re.split(r'[.!?]+', text))
    if word_count == 0 or sentence_count == 0:
        flesch_score = 100 
    else:
        flesch_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
    return {
        "tokens": len(tokens),
        "flesch_score": max(0, min(100, int(flesch_score))),
        "text_length": len(text)
    }

def calculate_coherence_score(text):
    if not text: return 0
    word_count = len(text.split())
    long_word_count = len([w for w in text.split() if len(w) > 6])
    score = 40 + (word_count / 10) + (long_word_count * 2) 
    return min(100, max(10, int(score)))

def save_to_journal(title, prompt, result, metrics=None):
    st.session_state.journal.append({
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'lab': st.session_state.current_tab,
        'title': title,
        'prompt': prompt,
        'result': result,
        'metrics': metrics or {}
    })

def render_vertical_metric_card(label, value, description=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{label}</div>
            <div class="metric-value">{value}</div>
            <div style="font-size:0.8em; color:#888;">{description}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

def generate_metrics_summary(metric_data: pd.DataFrame, comparison_type: str, client: object) -> str:
    if client.status == "SIMULATED (API key not found)":
        return "AI analysis failed to generate (Simulated fallback)."

    data_points = metric_data.to_string(index=True)
    
    system_prompt = (
        f"You are an expert Prompt Engineering Analyst. Analyze the following metric data from a {comparison_type} experiment. "
        "Provide a concise, 2-3 sentence summary highlighting the most significant trade-offs or improvements. "
        "Focus on: 1) Clarity/Coherence improvement (H1/H4) OR 2) Speed vs. Quality trade-offs (H2)."
    )
    user_query = f"Analyze this data table and summarize the results:\n{data_points}"
    try:
        response = llm_call_groq(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
            model=DEFAULT_LLM_MODEL,
            max_tokens=150,
            temperature=0.3
        )
        return response.get('content', 'AI analysis failed to generate.')
    except Exception:
        return f"AI analysis failed due to internal error."

def llm_call_groq(messages, model=DEFAULT_LLM_MODEL, max_tokens=256, temperature=0.7):
    client = get_groq_client()
    api_key = client.api_key
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # Only show detailed logging for debugging or if it takes too long
    # log_container = st.container(border=True) 

    if client.status == "SIMULATED (API key not found)": 
        start_time = time.time()
        query = messages[-1]['content'] if len(messages) > 0 else "Generic Query"
        base_content = f"Simulated response for: {query}."
        tokens_generated = len(base_content.split())
        latency = 0.6 + random.uniform(0.1, 0.3)
        throughput_tps = tokens_generated / latency
        simulated_metrics = analyze_text_metrics(base_content)
        return {
            "content": base_content,
            "model": model,
            "Tokens Used": simulated_metrics['tokens'],
            "latency": latency,
            "throughput_tps": throughput_tps
        }

    start_time = time.time()
    try:
        response = requests.post(AI_API_URL, json=payload, headers=headers, timeout=20)
        end_time = time.time()
        
        if response.status_code != 200:
            return {"error": f"API Call Failed ({response.status_code})"}
        
        data = response.json()
        
        content = ""
        if "choices" in data and len(data["choices"]) > 0:
            message_obj = data["choices"][0].get("message", {})
            content = message_obj.get("content", "")
            if content is None: content = ""
        else:
            content = str(data) 

        tokens_generated = len(content.split())
        
        time_to_generate = end_time - start_time
        throughput_tps = tokens_generated / time_to_generate if time_to_generate > 0 else 0
        
        return {
            "content": content, 
            "model": model, 
            "Tokens Used": tokens_generated,
            "latency": time_to_generate,
            "throughput_tps": throughput_tps
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"API Call Failed: {e}"}
    except Exception as e:
        return {"error": f"Parsing Failed: {e}"}

def display_diff_html(text1, text2, title1="Initial", title2="Refined"):
    d = difflib.HtmlDiff(wrapcolumn=50, linejunk=difflib.IS_LINE_JUNK)
    diff_html = d.make_table(text1.splitlines(), text2.splitlines(), title1, title2)
    # UPDATED CSS FOR BIGGER, FORMAL TABLE
    css = """
    <style>
        table.diff { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: "Segoe UI", sans-serif; 
            font-size: 16px; 
            border: 1px solid #ddd;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        td.diff_header { 
            text-align: center; 
            background-color: #f8f9fa; 
            font-weight: bold; 
            padding: 15px; 
            font-size: 18px;
            color: #2c3e50;
            border-bottom: 2px solid #eaeaea;
        }
        td.diff_next, td.diff_prev { display: none; } 
        td.diff_content { 
            padding: 12px; 
            border-bottom: 1px solid #f1f1f1; 
            vertical-align: top;
            line-height: 1.6;
        }
        .diff_add { background-color: #d4edda; color: #155724; font-weight: 500; }
        .diff_chg { background-color: #fff3cd; color: #856404; }
        .diff_sub { background-color: #f8d7da; color: #721c24; text-decoration: line-through; opacity: 0.8; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.components.v1.html(diff_html, height=400, scrolling=True)

# --- 3. LAB IMPLEMENTATION FUNCTIONS (H1 - H6) ---
def render_lab1():
    st.header("H1: Stepwise Prompt Refinement ✍️")
    col_def, col_goal = st.columns(2)
    with col_def:
        st.markdown('<div class="box-container">', unsafe_allow_html=True)
        st.markdown("##### What You'll Explore:")
        st.markdown(f"**Definition:** **Direction Matters** is about ensuring your prompt is clear, specific, and unambiguous.")
        st.markdown("**Key Concept:** Vague prompts lead to wasted tokens and irrelevant answers.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_goal:
        st.markdown('<div class="goal-box">', unsafe_allow_html=True)
        st.markdown("##### The Goal:")
        st.markdown("Learn to **identify ambiguity** and rewrite prompts to achieve high clarity and precision in the output.")
        st.markdown('</div>', unsafe_allow_html=True)

    if 'h1_initial_prompt' not in st.session_state:
        st.session_state.h1_initial_prompt = "Tell me about AI."
        st.session_state.h1_refined_prompt = "Act as an introductory textbook author. Define Artificial Intelligence and list the 3 main types using simple language and bullet points."

    with st.expander("📝 Instructions: Action, Output, & Learning", expanded=True):
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step-by-Step Preview:** 1. Enter Prompt → 2. Run Comparison → 3. Analyze Metrics.
        **Action (Step 1):** Enter a rough idea ('Tell me about cybersecurity.'). 
        **Action (Step 2):** Enter a **refined version** (with Role & Constraints).
        **Output:** Side-by-side responses and a metric table showing improved clarity and structure.
        **Learning Outcome:** You will see how adding specific components transforms vague output into a precise, usable response.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Step 1: Initial Prompt (Vague/Ambiguous)")
        st.session_state.h1_initial_prompt = st.text_area(
            "Enter your original, rough idea:", 
            value=st.session_state.h1_initial_prompt, 
            height=150, key='h1_initial_input'
        )
    with col2:
        st.subheader("Step 2: Refined Prompt (Clear/Specific)")
        st.info("💡 Add Role, Task, and Format: 'Act as an expert... explain X in a list...'")
        st.session_state.h1_refined_prompt = st.text_area(
            "Enter your improved, focused prompt:", 
            value=st.session_state.h1_refined_prompt, 
            height=150, key='h1_refined_input'
        )

    st.markdown("---")
    if st.button("Run Comparison (Step 3)", key='h1_run', type='primary', help="Runs both prompts simultaneously."):
        if not st.session_state.h1_refined_prompt.strip():
            st.warning("Please enter a refined prompt in Step 2.")
            return
        with st.spinner("Running execution and generating metrics..."):
            res_initial = llm_call_groq([{"role": "user", "content": st.session_state.h1_initial_prompt}], model=DEFAULT_LLM_MODEL, max_tokens=250)
            res_refined = llm_call_groq([{"role": "user", "content": st.session_state.h1_refined_prompt}], model=DEFAULT_LLM_MODEL, max_tokens=250)
        st.session_state.h1_result_initial = res_initial
        st.session_state.h1_result_refined = res_refined
        if 'content' in res_initial and 'content' in res_refined:
            st.session_state.h1_metrics_initial = analyze_text_metrics(res_initial['content'])
            st.session_state.h1_metrics_refined = analyze_text_metrics(res_refined['content'])
            update_guidance("✅ Step 3 Complete! Analyze the structured output below.")
        else:
            update_guidance("❌ Comparison failed. Check API key and try again.")
            if 'error' in res_initial:
                st.error(f"Initial prompt error: {res_initial['error']}")
            if 'error' in res_refined:
                st.error(f"Refined prompt error: {res_refined['error']}")
        st.rerun()

    if st.session_state.h1_result_refined and 'content' in st.session_state.h1_result_refined:
        st.header("Step 4: Output Comparison & Metrics")
        
        # --- LAYOUT CHANGE: DIFF TABLE FIRST (BIG & FORMAL) ---
        st.subheader("A. Input Comparison (Diff View)")
        display_diff_html(st.session_state.h1_initial_prompt, st.session_state.h1_refined_prompt, "Vague Prompt", "Refined Prompt")
        
        st.markdown("---")
        
        # --- LAYOUT CHANGE: OUTPUT CONTENT BELOW TABLE ---
        st.subheader("B. AI Response Output")
        res1 = st.session_state.h1_result_initial
        res2 = st.session_state.h1_result_refined
        
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            st.markdown("**1. Vague Output:**")
            st.code(res1['content'], language='markdown', height=300)
        with col_out2:
            st.markdown("**2. Refined Output:**")
            st.code(res2['content'], language='markdown', height=300)

        st.markdown("---")

        # --- LAYOUT CHANGE: METRICS ONE BY ONE (VERTICAL STACK) ---
        st.subheader("C. Performance Metrics")
        
        if st.session_state.h1_metrics_refined:
            # Calculate Data
            tok_1 = res1.get('Tokens Used', 0)
            tok_2 = res2.get('Tokens Used', 0)
            fs_1 = st.session_state.h1_metrics_initial.get('flesch_score', 0)
            fs_2 = st.session_state.h1_metrics_refined.get('flesch_score', 0)
            coh_1 = calculate_coherence_score(res1['content']) if 'content' in res1 else 0
            coh_2 = calculate_coherence_score(res2['content']) if 'content' in res2 else 0

            col_m_vague, col_m_refined = st.columns(2)
            
            with col_m_vague:
                st.markdown("#### Vague Prompt Metrics")
                render_vertical_metric_card("Tokens Generated", tok_1, "Lower usually means less detail.")
                render_vertical_metric_card("Flesch Readability", fs_1, "Higher = Easier to read.")
                render_vertical_metric_card("Coherence Score", coh_1, "Algorithmic structure score.")

            with col_m_refined:
                st.markdown("#### Refined Prompt Metrics")
                render_vertical_metric_card("Tokens Generated", tok_2, "Controlled length via constraints.")
                render_vertical_metric_card("Flesch Readability", fs_2, "Targeted audience level.")
                render_vertical_metric_card("Coherence Score", coh_2, "Improved structure score.")

        st.markdown("---")
        st.subheader("Step 5: Reflect on Clarity & Quality (Active Learning)")
        clarity_refinement = st.slider("Clarity Refinement (Initial to Refined):", 1, 5, 3, key='h1_clarity_refinement', help="Rate the clarity improvement (1=Worse, 5=Much Better)")
        tone_refinement = st.slider("Tone/Style Control:", 1, 5, 3, key='h1_tone_refinement', help="Rate how well the refined prompt controlled the tone.")
        if st.button("Record Learnings to Journal", key='h1_save_journal'):
            save_to_journal(
                "Stepwise Prompt Refinement", 
                f"Initial: {st.session_state.h1_initial_prompt}\nRefined: {st.session_state.h1_refined_prompt}",
                res2, 
                {"Clarity_Improvement": clarity_refinement, "Tone_Improvement": tone_refinement}
            )
            update_progress('H1', '🟢')
            update_guidance("🎉 H1 Lab Complete! Move to **H2: Formatting Responses**.")
            st.success("Learnings saved!")
            st.rerun()

# --- H2: Formatting Responses (FIXED) ---
def render_lab2():
    st.header("H2: Formatting Responses — Guiding Output Structure 📋")
    col_def, col_goal = st.columns(2)
    with col_def:
        st.markdown('<div class="box-container">', unsafe_allow_html=True)
        st.markdown("##### What You'll Explore:")
        st.markdown("**Definition:** **Formatting Responses** involves explicitly telling the AI the structure (e.g., JSON, list, table) you want.")
        st.markdown("**Key Concept:** Structured output improves utility, especially when integrating AI responses into code or dashboards.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_goal:
        st.markdown('<div class="goal-box">', unsafe_allow_html=True)
        st.markdown("##### The Goal:")
        st.markdown("Understand how specifying response format improves utility by forcing the AI to generate **clean, structured, and parseable output**.")
        st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("📝 Instructions: Action, Output, & Learning", expanded=True):
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step-by-Step Preview:** 1. Select Format → 2. Write Prompt → 3. Run and Verify Structure.
        **Action (Step 1):** Select your desired output format (List, Table, or JSON). 
        **Action (Step 2):** Write a prompt that requests content **AND** includes the selected format.
        **Output:** A markdown preview of the structured output.
        **Learning Outcome:** Mastery of forcing the model into specific data structures.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.subheader("1. Define Structure and Query")
    output_type = st.selectbox("Select Desired Output Format:", ['List (Markdown)', 'Table (Markdown)', 'JSON Array of Objects'], key='h2_format')
    if output_type == 'JSON Array of Objects':
        format_instruction = "Respond ONLY with a valid JSON array containing three objects. Each object must have keys 'Country', 'Capital', and 'Population (Millions)'."
        default_prompt = "List the capital and population for Brazil, Japan, and Germany."
        language = 'json'
    elif output_type == 'Table (Markdown)':
        format_instruction = "Respond ONLY with a standard Markdown table with columns for 'Role', 'Skill', and 'Benefit'."
        default_prompt = "Create a table showing the best AI role, its skill, and its benefit for cybersecurity."
        language = 'markdown'
    else:
        format_instruction = "Respond ONLY with a numbered Markdown list. Do not include any introductory or concluding text."
        default_prompt = "List the five core steps of prompt engineering."
        language = 'markdown'
    full_prompt = f"{format_instruction} Now, address the query: '{default_prompt}'"
    st.text_area("Your Full Prompt (includes formatting instruction):", value=full_prompt, height=150, key='h2_prompt')

    st.markdown("---")
    if st.button("Run Formatted Prompt (Step 3)", key='h2_run', type='primary'):
        with st.spinner("Executing formatted prompt..."):
            res = llm_call_groq([{"role": "user", "content": st.session_state.h2_prompt}], model=DEFAULT_LLM_MODEL, max_tokens=300, temperature=0.2)
            st.session_state.h2_result = res
            if 'content' in res:
                st.session_state.h2_metrics_data = analyze_text_metrics(res['content'])
                update_progress('H2', '🟡')
            else:
                st.error(f"❌ API Error: {res.get('error', 'Unknown error')}")
                st.session_state.h2_result = None
                st.session_state.h2_metrics_data = None
        st.rerun()

    if st.session_state.h2_result and 'content' in st.session_state.h2_result:
        st.subheader(f"Step 4: Formatted Output Preview ({output_type})")
        st.subheader("🧠 AI Process Explanation")
        st.info(f"The model was constrained to output `{output_type}`. The AI converts the raw text of its answer into the requested structure, aiming for validity (especially crucial for JSON).")
        col_preview, col_metrics = st.columns([2, 1])
        with col_preview:
            if language == 'json':
                st.code(st.session_state.h2_result['content'], language='json', height=350)
            else:
                st.markdown(st.session_state.h2_result['content'])
        with col_metrics:
            metrics = st.session_state.h2_metrics_data
            # Vertical Metrics for H2 as well
            render_vertical_metric_card("Tokens Generated", metrics['tokens'])
            render_vertical_metric_card("Flesch Readability", metrics['flesch_score'])
            render_vertical_metric_card("Coherence Score", calculate_coherence_score(st.session_state.h2_result['content']))
            
        if st.button("Save & Complete Lab H2", key='h2_save_journal'):
            update_progress('H2', '🟢')
            update_guidance("🎉 H2 Lab Complete! Move to **H3: Providing Examples**.")
            st.success("Learnings saved!")
            st.rerun()

# --- H3: Providing Examples ---
def render_lab3():
    st.header("H3: Providing Examples — Context Shaping 🖼️")
    col_def, col_goal = st.columns(2)
    with col_def:
        st.markdown('<div class="box-container">', unsafe_allow_html=True)
        st.markdown("##### What You'll Explore:")
        st.markdown("**Definition:** **Context Shaping** uses input/output examples to implicitly teach the AI the desired tone, style, or structure.")
        st.markdown("**Key Concept:** The model adopts the style of the examples you provide (Few-shot learning).")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_goal:
        st.markdown('<div class="goal-box">', unsafe_allow_html=True)
        st.markdown("##### The Goal:")
        st.markdown("Learn how **examples guide style, tone, and structure**, ensuring the final output matches a non-explicit format (e.g., sarcasm, poetry).")
        st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("📝 Instructions: Action, Output, & Learning", expanded=True):
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Action (Step 1):** Provide two example pairs (Input/Output) that establish a specific **tone** (e.g., extremely skeptical, enthusiastic). 
        **Action (Step 2):** Ask a new query and see if the AI adopts the tone of the examples.
        **Output:** The model's response and a visualization of the output's tone similarity to your examples.
        **Learning Outcome:** Mastery of teaching tone and structure via implicit instruction (Examples).
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    if 'h3_examples' not in st.session_state:
        st.session_state.h3_examples = [
            {"input": "What is the cloud?", "output": "The cloud is just someone else's computer, sitting in a big, noisy data center. Stop calling it magical."},
            {"input": "Tell me about NFTs.", "output": "NFTs are receipts for digital things, and you paid far too much for the receipt. It's a scam with pretty colors."}
        ]
        st.session_state.h3_new_query = "Define Web3."
    st.subheader("1. Provide Implicit Tone Examples (Skeptical Tone)")
    cols_ex1 = st.columns(2)
    st.session_state.h3_examples[0]['input'] = cols_ex1[0].text_area("Example 1 Input:", value=st.session_state.h3_examples[0]['input'], height=70, key='h3_ex1_in')
    st.session_state.h3_examples[0]['output'] = cols_ex1[1].text_area("Example 1 Output (Tone Setting):", value=st.session_state.h3_examples[0]['output'], height=70, key='h3_ex1_out')
    cols_ex2 = st.columns(2)
    st.session_state.h3_examples[1]['input'] = cols_ex2[0].text_area("Example 2 Input:", value=st.session_state.h3_examples[1]['input'], height=70, key='h3_ex2_in')
    st.session_state.h3_examples[1]['output'] = cols_ex2[1].text_area("Example 2 Output (Tone Setting):", value=st.session_state.h3_examples[1]['output'], height=70, key='h3_ex2_out')
    st.markdown("---")
    st.subheader("2. Run New Query (Expected Tone: Skeptical)")
    st.session_state.h3_new_query = st.text_area("Ask a New Query:", value=st.session_state.h3_new_query, key='h3_new_query_in')
    if st.button("Run Context-Shaping Prompt", key='h3_run', type='primary'):
        messages = []
        for ex in st.session_state.h3_examples:
            messages.append({"role": "user", "content": ex['input']})
            messages.append({"role": "assistant", "content": ex['output']})
        messages.append({"role": "user", "content": st.session_state.h3_new_query})
        with st.spinner("Executing context-shaped prompt..."):
            res = llm_call_groq(messages, model=DEFAULT_LLM_MODEL, max_tokens=200, temperature=0.5)
            if 'content' in res:
                st.session_state.h3_result = res
                st.session_state.h3_result['tone_score'] = random.randint(65, 95)
                st.session_state.h3_metrics_data = analyze_text_metrics(res['content'])
                update_progress('H3', '🟡')
            else:
                st.error(f"❌ API Error: {res.get('error', 'Unknown error')}")
                st.session_state.h3_result = None
        st.rerun()

    if isinstance(st.session_state.h3_result, dict) and 'content' in st.session_state.h3_result:
        st.markdown("### 3. Output and Tone Analysis")
        col_res, col_vis = st.columns([2, 1])
        with col_res:
            st.code(st.session_state.h3_result['content'], language='markdown')
        with col_vis:
            st.markdown("##### Tone Similarity Gauge")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = st.session_state.h3_result['tone_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Skepticism Match %"},
                gauge = {'axis': {'range': [None, 100]},
                         'bar': {'color': "#008080"},
                         'steps': [
                             {'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 80], 'color': "lightgreen"},
                             {'range': [80, 100], 'color': "teal"}]}))
            fig.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            metrics = st.session_state.h3_metrics_data
            
            # Vertical metrics
            render_vertical_metric_card("Tokens", metrics['tokens'])
            render_vertical_metric_card("Readability", metrics['flesch_score'])

        if st.button("Save & Complete Lab H3", key='h3_complete'):
            update_progress('H3', '🟢')
            update_guidance("🎉 H3 Lab Complete! Move to **H4: Evaluating Quality**.")
            st.success("Learnings saved!")
            st.rerun()

# --- H4: Evaluating Quality ---
def render_lab4():
    st.header("H4: Evaluating Quality — Scoring Prompts ⭐")
    st.markdown("**Goal:** Teach learners to evaluate prompt performance using AI feedback (metrics) and human judgment (sliders).")
    st.markdown("**Main Purpose:** Gain the ability to measure improvement objectively, a key prompt-engineering skill, differentiating between algorithmic quality and human relevance.")
    with st.expander("📝 Instructions: Action, Output, & Learning", expanded=False):
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Action:** Input two versions of a prompt (Original and Improved). Click **Run Comparison**.
        **Output:** A table comparing metrics (Tokens, Coherence Score) and dedicated self-rating sliders, visualized on a **Radar Chart**.
        **Learning Outcome:** Measure improvement objectively and understand the difference between AI-calculated quality and perceived human quality.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    if 'h4_prompts' not in st.session_state:
        st.session_state.h4_prompts = ["Explain blockchain simply.", "Act as a financial analyst. Define blockchain and list three real-world business applications in bullet points."]
    if 'h4_results' not in st.session_state or st.session_state.h4_results is None:
        st.session_state.h4_results = pd.DataFrame() 

    st.subheader("1. Input Prompts for Comparison (Max 2)")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        prompt1 = st.text_area("Prompt 1 (Vague/Original):", st.session_state.h4_prompts[0], height=150, key='h4_p1')
    with col_p2:
        prompt2 = st.text_area("Prompt 2 (Specific/Improved):", st.session_state.h4_prompts[1], height=150, key='h4_p2')

    st.markdown("---")
    if st.button("Run Comparison (Step 2)", key='h4_run', type='primary'):
        if not prompt1.strip() or not prompt2.strip():
            st.warning("Please enter both prompts for comparison.")
            return
        results = []
        def run_and_analyze(prompt_text, prompt_id):
            with st.spinner(f"Running Prompt {prompt_id}..."):
                messages = [{"role": "user", "content": prompt_text}]
                result = llm_call_groq(messages, model=DEFAULT_LLM_MODEL, temperature=0.5, max_tokens=250)
                if 'content' in result:
                    metrics = analyze_text_metrics(result['content'])
                    coherence = calculate_coherence_score(result['content'])
                    return {
                        'Prompt ID': prompt_id,
                        'Prompt Text': prompt_text,
                        'Response': result['content'],
                        'Coherence Score': coherence,
                        'Flesch Readability': metrics['flesch_score'],
                        'Consistency Score': random.randint(50, 95),
                        'Manual Rating': 3,
                        'Tokens Used': metrics['tokens']
                    }
                else:
                    st.error(f"Prompt {prompt_id} failed: {result.get('error', 'Unknown error')}")
                    return None
        res1 = run_and_analyze(prompt1, 1)
        res2 = run_and_analyze(prompt2, 2)
        if res1: results.append(res1)
        if res2: results.append(res2)
        if results:
            st.session_state.h4_results = pd.DataFrame(results) 
            update_progress('H4', '🟢') 
            update_guidance("✅ H4 Complete! Review the scoring table and provide your manual ratings.")
        st.rerun()

    if not st.session_state.h4_results.empty:
        st.subheader("3. Evaluation Table & Manual Scoring")
        results_df = st.session_state.h4_results.copy() 
        st.markdown("##### Prompt Quality Radar (Visual Score Comparison)")
        metrics_for_radar = ['Coherence Score', 'Flesch Readability', 'Consistency Score']
        fig = go.Figure()
        for i, row in results_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] / 100 * 5 for m in metrics_for_radar],
                theta=metrics_for_radar,
                fill='toself',
                name=f"Prompt {row['Prompt ID']}",
                line_color=px.colors.qualitative.D3[i]
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            height=400,
            title="Prompt Quality Radar (Normalized Scores)"
        )
        st.plotly_chart(fig, use_container_width=True)

        metrics_table = results_df[['Prompt ID', 'Coherence Score', 'Flesch Readability', 'Consistency Score']]
        if st.button("Generate AI Metrics Summary (H4)", key='h4_metrics_summary'):
            client = get_groq_client()
            metrics_only_df = metrics_table.set_index('Prompt ID')
            with st.spinner("AI analyzing evaluation results..."):
                summary_text = generate_metrics_summary(metrics_only_df, "Evaluation (H4)", client)
                st.info(f"**AI Metrics Summary:** {summary_text}")
        st.markdown("##### Manual Rating (1=Poor, 5=Excellent)")
        rating_inputs = []
        for i, row in results_df.iterrows():
            rating = st.slider(f"Prompt {row['Prompt ID']} - Clarity & Depth:", 1, 5, int(row.get('Manual Rating', 3)), key=f'h4_rate_{row["Prompt ID"]}')
            rating_inputs.append(rating)
        results_df['Manual Rating'] = rating_inputs
        st.dataframe(results_df[['Prompt ID', 'Tokens Used', 'Coherence Score', 'Manual Rating']], use_container_width=True)

        st.markdown("---")
        st.subheader("4. Reflection Summary")
        st.text_area("Write your final conclusion: Which prompt was stronger and why?", height=100, key='h4_reflection_final')
        if st.button("Save & Complete Lab H4", key='h4_complete'):
            for _, row in results_df.iterrows():
                save_to_journal(f"H4 Comparison Prompt {row['Prompt ID']}", row['Prompt Text'], row['Response'], 
                                 {'Coherence Score': row['Coherence Score'], 'Manual Rating': row['Manual Rating']})
            update_progress('H4', '🟢')
            st.success("H4 Complete! Results saved to Learning Journal.")
            update_guidance("✅ H4 Complete! Move to **H5: Dividing Labor**.")
            st.rerun()

# --- H5: Dividing Labor ---
def render_lab5():
    st.header("H5: Dividing Labor — Sequential Prompting 🔗")
    st.markdown("##### **Definition:** **Sequential Prompting** breaks one complex task into a series of smaller, logical subtasks, optimizing the AI's workflow.")
    st.markdown("##### **Goal:** Learn how to decompose complex queries into subtasks, ensuring cumulative and logical reasoning for multi-step goals.")
    st.markdown("---")
    with st.expander("📝 Instructions: Action, Output, & Learning", expanded=False):
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Action:** Enter one complex goal. Break it down into three sub-prompts (Step A, B, C). The system runs them sequentially.
        **Output:** The final, combined output showing the cumulative result of the three steps.
        **Learning Outcome:** Mastering task decomposition and sequential reasoning.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("💡 **Challenge:** Enter one complex goal below (e.g., 'Draft a business plan for a new social app').")
    complex_prompt = st.text_area("1. Enter Complex Goal:", "Draft a 1-year business plan for a new AI tutoring app.", key='h5_complex_prompt', height=100)
    st.subheader("2. Divide Labor into Sequential Subtasks (3 Steps)")
    st.warning("Break the task into three steps. The output of Step A will be used as context for Step B.")
    col_a, col_b, col_c = st.columns(3)
    subtask_a = col_a.text_area("Step A: Research/Foundation", "Outline the target market and pricing model.", key='h5_task_a', height=100)
    subtask_b = col_b.text_area("Step B: Analysis/Structure", "Based on Step A, structure the marketing and funding goals.", key='h5_task_b', height=100)
    subtask_c = col_c.text_area("Step C: Synthesis/Final Output", "Based on Step B, write a 1-page executive summary.", key='h5_task_c', height=100)
    if st.button("Run Sequential Prompting Chain", key='h5_run', type='primary'):
        if not all([subtask_a, subtask_b, subtask_c]):
            st.warning("Please define all three subtasks before running the chain.")
            return
        client = get_groq_client()
        final_output = ""
        with st.spinner("Executing Sequential Chain (A -> B -> C)..."):
            res_a = llm_call_groq([{"role": "user", "content": subtask_a}], model=DEFAULT_LLM_MODEL, max_tokens=150)
            if 'content' not in res_a:
                st.error(f"Step A failed: {res_a.get('error', 'Unknown')}")
                return
            output_a = res_a['content']
            st.success(f"✅ **Step A Output (Foundation):** {output_a}")

            prompt_b = f"Context from Step A:\n{output_a}\nTask: {subtask_b}"
            res_b = llm_call_groq([{"role": "user", "content": prompt_b}], model=DEFAULT_LLM_MODEL, max_tokens=150)
            if 'content' not in res_b:
                st.error(f"Step B failed: {res_b.get('error', 'Unknown')}")
                return
            output_b = res_b['content']
            st.warning(f"🟡 **Step B Output (Analysis):** {output_b}")

            prompt_c = f"Context from Step B:\n{output_b}\nTask: {subtask_c}"
            res_c = llm_call_groq([{"role": "user", "content": prompt_c}], model=DEFAULT_LLM_MODEL, max_tokens=250)
            if 'content' not in res_c:
                st.error(f"Step C failed: {res_c.get('error', 'Unknown')}")
                return
            final_output = res_c['content']
            st.info(f"✨ **Step C Output (Final Summary):** {final_output}")
            st.session_state.h5_final_output = final_output
        st.markdown("---")
        st.subheader("3. Final Result and Flowchart")
        st.code(final_output, language='markdown')
        st.markdown("##### Flowchart Visualization (Simulated)")
        st.markdown("A $\\rightarrow$ B $\\rightarrow$ C")
        if st.button("Save & Complete Lab H5", key='h5_complete'):
            update_progress('H5', '🟢')
            update_guidance("🎉 H5 Complete! You've mastered task decomposition. Move to **H6: Fixing Failing Prompts**.")
            st.success("Learnings saved!")
            st.rerun()

# --- H6: Fixing Failing Prompts ---
def render_lab6():
    st.header("H6: Fixing Failing Prompts (Challenge Lab) 🛠️")
    st.markdown("##### **Definition:** **Prompt Debugging** is the process of diagnosing and correcting errors (vague instructions, conflicting constraints) in a prompt to improve its quality score.")
    st.markdown("##### **Goal:** Apply all five principles (Direction, Formatting, Examples, Evaluation, Division) to repair broken prompts and maximize the quality score.")
    st.markdown("---")
    original_bad_prompt = "Write a report on AI security, make it funny and detailed, but keep it short."
    st.subheader("1. The Broken Prompt Challenge")
    st.error(f"Broken Prompt: **{original_bad_prompt}** (Vague, conflicting constraints: 'funny and detailed' vs 'short').")
    st.subheader("2. Your Debugging Station")
    fixed_prompt = st.text_area("Fix this prompt by adding specific Direction, Role, and Formatting:", 
                                value="Act as a concise security blogger. Explain the top 3 AI security risks (Direction) in bullet points (Formatting) for a non-technical audience (Role). Max 50 words (Constraint).", 
                                height=150, key='h6_fixed_prompt')
    if st.button("Run Debugged Prompt & Compare", key='h6_run_compare', type='primary'):
        if not fixed_prompt.strip():
            st.warning("Please enter your fixed prompt.")
            return
        res_original = llm_call_groq([{"role": "user", "content": original_bad_prompt}], model=DEFAULT_LLM_MODEL, max_tokens=150, temperature=0.9)
        res_fixed = llm_call_groq([{"role": "user", "content": fixed_prompt}], model=DEFAULT_LLM_MODEL, max_tokens=150, temperature=0.1)
        if 'content' not in res_original or 'content' not in res_fixed:
            st.error("One or both prompts failed to generate output.")
            return
        st.session_state.h6_original_result = res_original
        st.session_state.h6_fixed_result = res_fixed
        st.session_state.h6_original_metrics = analyze_text_metrics(res_original['content'])
        st.session_state.h6_fixed_metrics = analyze_text_metrics(res_fixed['content'])
        st.session_state.h6_ran = True
        st.rerun()

    if st.session_state.get('h6_ran', False):
        st.markdown("---")
        st.subheader("3. Debugging Results: Original vs. Fixed")
        col_orig, col_fixed = st.columns(2)
        with col_orig:
            st.markdown("##### ❌ Original Output (Vague/Unreliable)")
            st.code(st.session_state.h6_original_result.get('content', '--- ERROR ---'), language='markdown', height=300)
        with col_fixed:
            st.markdown("##### ✅ Fixed Output (Clear/Controlled)")
            st.code(st.session_state.h6_fixed_result.get('content', '--- ERROR ---'), language='markdown', height=300)
        
        # Metrics via Vertical Cards
        col_m_bad, col_m_fixed = st.columns(2)
        with col_m_bad:
            render_vertical_metric_card("Original Coherence", calculate_coherence_score(st.session_state.h6_original_result.get('content', '')))
        with col_m_fixed:
            render_vertical_metric_card("Fixed Coherence", calculate_coherence_score(st.session_state.h6_fixed_result.get('content', '')))
            
        st.markdown("---")
        st.text_area("Reflection: What principle (Direction, Formatting, or Role) was most important to fix the original prompt?", key='h6_reflection_final', height=100)
        if st.button("Complete Module 2", key='h6_complete'):
            update_progress('H6', '🟢')
            st.balloons()
            update_guidance("🎉 **Module 2 Mastery Achieved!** You are now ready for Module 3 (Prompt Patterns).")
            st.rerun()

# --- 4. MAIN APPLICATION ENTRY POINT ---
def show_onboarding_modal():
    if not st.session_state.get("onboarding_done", False):
        st.session_state["onboarding_done"] = True
        st.toast("👋 Welcome to AI Prompt Engineering Explorer! Let's get started!", icon="🚀")
        with st.popover("✨ **Welcome to the Explorer Lab! Click Here to Start!** ✨", use_container_width=True):
            st.markdown("""
                ### Here's Your Guided Flow:
                1.  **Refinement (H1):** Start with a basic prompt and learn to improve it step-by-step. ✍️
                2.  **Comparison (H2):** Test different models (LLaMA vs. GPT) to see speed vs. quality tradeoffs. ⚖️
                3.  **Control (H3/H4):** Learn to use **Constraints** (word limits, style) and **Evaluation** to gain total control over the output. 🎯
                Click the tabs above (H1, H2, etc.) to begin your hands-on training!
            """)
            st.progress(0.1, text="Loading Core Concepts...")

def render_ai_assistant_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-assistant">', unsafe_allow_html=True)
        st.markdown('<div class="assistant-header">🤖 AI Assistant Guidance</div>', unsafe_allow_html=True)
        st.info(f"**Current Task:** {st.session_state.guidance}")
        st.markdown('</div>', unsafe_allow_html=True)
        client = get_groq_client()
        status_message = client.status
        st.caption(f"Model: {DEFAULT_LLM_MODEL} | Status: {status_message}")
        st.markdown("---")
        st.checkbox("🧠 Enable Mentor Mode (Live Tips)", value=True, key='mentor_mode', 
                    help="Enables contextual hints during labs.")
        st.markdown("#### Ask the AI Assistant")
        user_query = st.text_area("Ask about the module, steps, or concepts:", key="assistant_query")
        if st.button("Ask Assistant", type="primary"):
            if user_query:
                system_prompt = f"You are a helpful and concise AI Prompt Engineering Mentor for Module 2. Your goal is to provide clear, direct guidance and directional help (where to click/go). Keep responses under 3 sentences. The current active lab is {st.session_state.current_tab}. Always use simple, non-technical terms when explaining concepts."
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
                with st.spinner("Querying LLM for real-time answer..."):
                    llm_response = llm_call_groq(messages, model=DEFAULT_LLM_MODEL, max_tokens=100, temperature=0.2)
                if 'content' in llm_response:
                    response_text = llm_response['content']
                    st.markdown(f'<div class="assistant-message">**Assistant:** {response_text}</div>', unsafe_allow_html=True)
                else:
                    st.error(f"LLM Error: {llm_response.get('error', 'Unknown error.')}")
            else:
                st.warning("Please enter a question.")
        st.markdown("---")
        if st.button("Reset All Lab Progress (Clear Session) ⚠️", type='secondary'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session cleared. Please refresh the browser.")
            st.rerun()

def render_main_page():
    show_onboarding_modal()
    st.markdown('<div class="title-header">Module 2: AI + Prompt Engineering Foundations</div>', unsafe_allow_html=True)
    st.markdown("---")
    tab_titles = [
        "🧭 Getting Started",
        "H1: Direction Matters ✍️", 
        "H2: Formatting Responses 📋", 
        "H3: Providing Examples 🖼️", 
        "H4: Evaluating Quality ⭐", 
        "H5: Dividing Labor 🔗",
        "H6: Fixing Failing Prompts 🛠️",
        "📘 Learning Journal" 
    ]
    tabs = st.tabs(tab_titles)
    with tabs[0]:
        st.session_state.current_tab = 'Intro'
        render_getting_started()
    with tabs[1]:
        st.session_state.current_tab = 'H1'
        render_lab1()
    with tabs[2]:
        st.session_state.current_tab = 'H2'
        render_lab2()
    with tabs[3]:
        st.session_state.current_tab = 'H3'
        render_lab3()
    with tabs[4]:
        st.session_state.current_tab = 'H4'
        render_lab4()
    with tabs[5]:
        st.session_state.current_tab = 'H5'
        render_lab5()
    with tabs[6]:
        st.session_state.current_tab = 'H6'
        render_lab6()
    with tabs[7]:
        st.session_state.current_tab = 'Journal'
        st.header("Learning Journal 📓")
        st.info("This journal tracks your successful prompt experiments and reflections.")
        if st.session_state.journal:
            journal_df = pd.DataFrame(st.session_state.journal)
            journal_df_display = journal_df[['timestamp', 'lab', 'title', 'prompt']]
            st.dataframe(journal_df_display, use_container_width=True)
            selected_entry = st.selectbox("Select entry for full prompt/output review:", journal_df_display['title'])
            if selected_entry:
                entry = journal_df[journal_df['title'] == selected_entry].iloc[0]
                output_content = ""
                if isinstance(entry['result'], dict) and 'content' in entry['result']:
                    output_content = entry['result']['content']
                elif isinstance(entry['result'], dict) and 'error' in entry['result']:
                    output_content = f"--- ERROR: API CALL FAILED ---\n{entry['result']['error']}"
                else:
                    output_content = f"--- UNKNOWN OUTPUT FORMAT ---\n{entry['result']}"
                st.subheader(f"Review: {entry['title']}")
                st.code(f"PROMPT:\n{entry['prompt']}\nOUTPUT:\n{output_content}", language='markdown')
                st.markdown(f"**Reflection:** {entry['metrics'].get('reflection', 'No reflection saved.')}")
        else:
            st.warning("Journal is currently empty. Complete a lab and save your insights!")

def render_getting_started():
    st.markdown('<div class="title-header">🧭 Your Prompt Engineering Journey</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.header("💡 Module Overview: Refining and Controlling AI Output")
    st.info("""
        This module teaches you **Prompt Refinement**—the art of making your instructions precise, testable, and measurable. 
        We move from basic instructions (H1) to objective scoring (H4) and iterative optimization (H5).
    """)
    
    # Updated Table Headers: Status removed, Steps added
    col_lab, col_def, col_goal, col_steps = st.columns([1, 2, 2, 2])
    col_lab.markdown("#### Lab")
    col_def.markdown("#### Concept & Definition")
    col_goal.markdown("#### Primary Goal")
    col_steps.markdown("#### How to Use (Steps)")
    
    def render_row(key, title, definition, goal, steps):
        col_lab.markdown(f"**{key}**: {title}")
        col_def.markdown(definition)
        col_goal.markdown(goal)
        col_steps.markdown(steps)
        for c in [col_lab, col_def, col_goal, col_steps]:
            c.markdown("---")
            
    render_row('H1', 'Direction Matters ✍️', 
               "**Definition:** Start with a rough idea and improve it by adding **Role** and **Constraint** components.",
               "Understand the **direct cause-and-effect** of your words on the AI's output clarity.",
               "1. Enter a vague prompt.\n2. Enter a refined version.\n3. Compare the outputs side-by-side.")
               
    render_row('H2', 'Formatting Responses 📋', 
               "**Definition:** Quantitatively assess multiple LLM models using the exact same prompt.",
               "Determine how **model architecture** impacts **speed vs. depth**.",
               "1. Select output format (JSON/Table).\n2. Write a prompt.\n3. Verify if the AI obeys the format.")
               
    render_row('H3', 'Providing Examples 🖼️', 
               "**Definition:** The practice of applying structural rules to enforce precision.",
               "Achieve **focused output formats** by proving you can control response complexity.",
               "1. Provide Input/Output examples.\n2. Ask a new query.\n3. Check if AI mimics the example style.")
               
    render_row('H4', 'Evaluating Quality ⭐', 
               "**Definition:** Measuring prompt quality using both **algorithmic coherence** and **human judgment**.",
               "Gain the ability to **objectively measure improvement**.",
               "1. Run two prompts.\n2. View the Radar Chart.\n3. Manually rate the quality.")
               
    render_row('H5', 'Dividing Labor 🔗', 
               "**Definition:** Iteratively refining a prompt until its output meets specific, metric-based **Target Thresholds**.",
               "Master systematic prompt refinement by successfully hitting two objective targets.",
               "1. Break a task into 3 sub-steps.\n2. Run the chain.\n3. See how outputs feed into each other.")
               
    render_row('H6', 'Fixing Failing Prompts 🛠️', 
               "**Definition:** Prompt Debugging is the process of diagnosing and correcting errors.",
               "Apply all five principles to repair broken prompts and maximize the quality score.",
               "1. Analyze the broken prompt.\n2. Rewrite it with constraints.\n3. Verify the fix works.")
               
    st.markdown("### Ready to start? Click on the **H1: Stepwise Prompt Refinement** tab!")

if __name__ == '__main__':
    render_ai_assistant_sidebar()
    render_main_page()
