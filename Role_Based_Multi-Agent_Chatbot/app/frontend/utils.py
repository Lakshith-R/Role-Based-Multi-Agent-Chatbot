import streamlit as st

def apply_custom_css():
    """Apply native Streamlit dark theme precisely matching the screenshot."""
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            /* ========== GLOBAL DARK THEME (NATIVE STREAMLIT MATCH) ========== */
            html, body, .stApp, [data-testid="stAppViewContainer"], 
            [data-testid="stHeader"], .main {
                background-color: #0e1117 !important;
                color: #fafafa !important;
            }
            
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            }
            
            /* Hide top Streamlit padding */
            .main .block-container {
                background-color: #0e1117 !important;
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
                max-width: 1500px !important;
            }



            /* Add this to utils.py style section */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: #161b22 !important; /* Slightly lighter than main background */
                border: 1px solid #31333f !important;
                border-radius: 8px !important;
                padding: 1rem !important;
                margin-bottom: 1rem !important;
            }

            /* This specifically targets the chat history area if you use height */
            [data-testid="stVerticalBlockBorderWrapper"] > div {
                background-color: transparent !important;
            }
            
            /* ========== SIDEBAR ========== */
            section[data-testid="stSidebar"] {
                background-color: #262730 !important;
                border-right: 1px solid #31333f !important;
            }
            
            section[data-testid="stSidebar"] > div {
                background-color: #262730 !important;
            }
            
            /* Sidebar Headers */
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                color: #ffffff !important;
                font-weight: 600 !important;
            }
            
            /* Sidebar Text */
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] div {
                color: #fafafa !important;
            }
            
            /* ========== METRICS ========== */
            [data-testid="stMetricValue"] {
                color: #ffffff !important;
                font-size: 1.2rem !important;
                font-weight: 700 !important;
            }
            
            [data-testid="stMetricLabel"] {
                color: #808495 !important;
                font-weight: 500 !important;
                text-transform: uppercase !important;
                font-size: 0.65rem !important;
            }
            
            /* Remove metric delta padding if any */
            [data-testid="stMetric"] {
                padding: 0 !important;
            }
            
            /* ========== CHAT MESSAGES ========== */
            [data-testid="stChatMessage"] {
                border-radius: 8px !important;
                padding: 0.8rem 1rem !important;
                margin-bottom: 0.5rem !important;
                background-color: #262730 !important;
                border: 1px solid #31333f !important;
            }
            
            /* User Messages */
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
                background-color: #1e3a5f !important;
                border-color: #3b82f6 !important;
            }
            
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div {
                color: #ffffff !important;
            }
            
            /* Assistant Messages */
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
                background-color: #262730 !important;
                border-color: #31333f !important;
            }
            
            /* ========== HEADERS ========== */
            h1, h2, h3, h4 {
                color: #ffffff !important;
            }
            
            /* ========== BUTTONS ========== */
            .stButton > button {
                background-color: #3b82f6 !important;
                color: #ffffff !important;
                border: none !important;
                font-size: 0.85rem !important;
                border-radius: 4px !important;
            }
            
            .stButton > button:hover {
                background-color: #2563eb !important;
                border-color: transparent !important;
            }
            
            /* ========== INPUT FIELDS ========== */
            .stTextInput input, .stChatInput textarea {
                border: 1px solid #31333f !important;
                background-color: #0e1117 !important;
                color: #fafafa !important;
                border-radius: 4px !important;
            }
            
            .stChatInput {
                width: 100% !important;
            }
            
            /* ========== EXPANDERS ========== */
            .streamlit-expanderHeader {
                background-color: #262730 !important;
                color: #ffffff !important;
                border: 1px solid #31333f !important;
            }
            
            .streamlit-expanderContent {
                background-color: #161b22 !important;
                border: 1px solid #31333f !important;
            }
            
            /* ========== DARK THEME TEXT ========== */
            .stMarkdown, .stMarkdown p, .stMarkdown span {
                color: #fafafa !important;
            }
            
            [data-testid="stVerticalBlock"] > [data-testid="element-container"] {
                margin-bottom: 0.25rem !important;
            }
        </style>
    """, unsafe_allow_html=True)
