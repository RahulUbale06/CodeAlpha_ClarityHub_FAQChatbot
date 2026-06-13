import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# PAGE CONFIG & INITIALIZATION
# ------------------------------
st.set_page_config(page_title="Clarity Hub", page_icon="⚡", layout="centered")

# Manage complex session states safely
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "selected_category" not in st.session_state:
    st.session_state.selected_category = "All Topics"


def set_search_state(query_text, category_text=None):
    st.session_state.search_query = query_text
    if category_text:
        st.session_state.selected_category = category_text


# ------------------------------
# THEME: ADVANCED NORDIC MIDNIGHT (CSS)
# ------------------------------
st.markdown(
    """
    <style>
    /* Global Container Theme Overrides */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0F172A !important; 
        color: #E2E8F0 !important; 
    }
    
    /* Sleek Typography styles */
    h1 {
        color: #06B6D4 !important; 
        font-weight: 800 !important;
        letter-spacing: -0.04em;
        margin-bottom: 5px !important;
    }
    
    /* Main Answer Spotlight Card */
    .faq-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 32px;
        border-radius: 20px;
        margin: 20px 0 10px 0;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        position: relative;
    }
    
    .faq-card::before {
        content: "TOP RESOLUTION";
        position: absolute;
        top: -10px;
        left: 20px;
        background: #06B6D4;
        color: #0F172A;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 3px 10px;
        border-radius: 6px;
    }
    
    .faq-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 14px;
        color: #38BDF8; 
    }
    
    .faq-body {
        font-size: 1.08rem;
        line-height: 1.75;
        color: #F8FAFC;
    }
    
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid #334155;
        font-size: 0.85rem;
        color: #64748B;
    }

    /* Style corrections for native Streamlit text boxes */
    input {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }
    input:focus {
        border-color: #06B6D4 !important;
        box-shadow: 0 0 0 1px #06B6D4 !important;
    }

    /* Target specific Streamlit structural containers */
    div[data-testid="stExpander"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
    }

    /* Custom Footer Attribution */
    .designer-footer {
        text-align: center;
        margin-top: 120px;
        padding-top: 24px;
        border-top: 1px solid #1E293B;
        font-size: 0.78rem;
        letter-spacing: 0.15em;
        color: #475569;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .designer-footer:hover {
        color: #06B6D4;
        letter-spacing: 0.18em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# ENTERPRISE DATA LOAD ENGINE
# ------------------------------
@st.cache_data(show_spinner=False)
def load_rich_dataset(path="faq.csv"):
    """Loads system data; provides high-fidelity default data if CSV does not exist."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Category" not in df.columns:
            df["Category"] = "General"
        return df

    # Custom structured fallback database
    mock_data = {
        "Category": [
            "Shipping",
            "Shipping",
            "Payments",
            "Payments",
            "Account",
            "Account",
        ],
        "Question": [
            "How can I track my package updates?",
            "Do you support international delivery locations?",
            "What methods of payment are securely handled?",
            "How long do refund settlements take to process?",
            "Where can I quickly change my account login password?",
            "How do I configure secondary Multi-Factor Authentication?",
        ],
        "Answer": [
            "We dispatch tracking notification tokens to your registered email address the moment freight leaves our hub operations.",
            "We handle delivery logic to over 60 global jurisdictions via courier networks with automated dynamic tariff computation.",
            "Our gateways safely authorize corporate transactions using major Credit providers, PayPal portals, and encrypted Apple Pay protocols.",
            "Approved refund transactions settle directly back to your original processing financial node within 5-7 standard operational business days.",
            "Navigate directly to User Configuration profiles, toggle the Credential Management column, and authorize a secure password reset sequence.",
            "Under Account Protection configurations, click 'Enable MFA' to generate an encrypted QR passkey for validation apps.",
        ],
    }
    return pd.DataFrame(mock_data)


@st.cache_resource(show_spinner=False)
def build_vector_matrix(questions_list):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(questions_list)
    return vectorizer, matrix


# Global calculations
df = load_rich_dataset()
categories = ["All Topics"] + sorted(df["Category"].unique().tolist())

# Filter view dynamically based on selected Category pill
if st.session_state.selected_category == "All Topics":
    filtered_df = df
else:
    filtered_df = df[df["Category"] == st.session_state.selected_category]

questions = filtered_df["Question"].tolist()
answers = filtered_df["Answer"].tolist()

# Train models contextually based on filtered questions
vectorizer, question_vectors = build_vector_matrix(questions)

# ------------------------------
# SYSTEM UI CONTAINER HEADER
# ------------------------------
st.title("Clarity")
st.write("Access instant organizational documentation answers via semantic search.")
st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------
# ADVANCED FEATURE 1: CATEGORY SELECTION PILLS
# ------------------------------
st.markdown(
    '<p class="suggestion-tip">📁 Filter Knowledge Base by Domain:</p>',
    unsafe_allow_html=True,
)
cat_cols = st.columns(len(categories))
for idx, col in enumerate(cat_cols):
    cat_name = categories[idx]
    # Highlight active category visually using native button variants
    is_active = cat_name == st.session_state.selected_category
    if col.button(
        f"{'🟢 ' if is_active else ''}{cat_name}",
        key=f"cat_{idx}",
        use_container_width=True,
    ):
        st.session_state.selected_category = cat_name
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------
# ADVANCED FEATURE 2: INPUT AND STATE MANAGEMENT RESET
# ------------------------------
search_col, clear_col = st.columns([5, 1])

with search_col:
    query_input = st.text_input(
        "Search Context Input Bar",
        value=st.session_state.search_query,
        label_visibility="collapsed",
        placeholder=f"Search inside {st.session_state.selected_category}... (e.g., tracking, validation, refund)",
    )

with clear_col:
    if st.button("Reset ✕", use_container_width=True):
        set_search_state("", "All Topics")
        st.rerun()

# ------------------------------
# SEARCH EXECUTION PIPELINE
# ------------------------------
has_query = query_input.strip() != ""

if has_query and len(questions) > 0:
    clean_query = query_input.strip()

    # Pass text through matching framework
    user_vec = vectorizer.transform([clean_query])
    scores = cosine_similarity(user_vec, question_vectors)[0]

    # Sort indices by match accuracy descending
    sorted_indices = scores.argsort()[::-1]
    best_match_idx = sorted_indices[0]
    top_confidence = float(scores[best_match_idx])

    # Check for character overlap if TF-IDF yields borderline confidence
    is_typo_fallback = False
    if top_confidence < 0.20:
        # Simple character intersection check to catch typos like "trcking" or "pword"
        query_set = set(clean_query.lower())
        for idx, q in enumerate(questions):
            intersection = len(query_set.intersection(set(q.lower())))
            if intersection / len(query_set) > 0.75:
                best_match_idx = idx
                top_confidence = 0.25  # Force fallback threshold override
                is_typo_fallback = True
                break

    # Presentation logic
    if top_confidence >= 0.20:
        # Render clean HTML Spotlight interface block
        st.markdown(
            f"""
            <div class="faq-card">
                <div class="faq-title">{questions[best_match_idx]}</div>
                <div class="faq-body">{answers[best_match_idx]}</div>
                <div class="card-footer">
                    <span>Domain: <b>{filtered_df.iloc[best_match_idx]['Category']}</b></span>
                    <span>Confidence Index: {int(top_confidence * 100)}% {'(Typo Assisted)' if is_typo_fallback else ''}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Micro-interaction engagement feedback engine
        feed_col1, feed_col2, feed_col3 = st.columns([3, 1, 1])
        with feed_col1:
            st.caption("Was this documentation engine entry complete?")
        with feed_col2:
            if st.button("👍 Useful", key="btn_yes", use_container_width=True):
                st.toast("Telemetry data logged. Thank you!", icon="✨")
        with feed_col3:
            if st.button("👎 Incomplete", key="btn_no", use_container_width=True):
                st.toast("Escalated to content team for revision.", icon="📝")

        # ------------------------------
        # ADVANCED FEATURE 3: RUNNER-UP RECOMMENDATIONS
        # ------------------------------
        if len(sorted_indices) > 1 and not is_typo_fallback:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("📋 Other related knowledge base entries:")

            # Show next 2 matches if they possess minor conversational relevance
            visible_counter = 0
            for runner_up_idx in sorted_indices[1:3]:
                runner_confidence = float(scores[runner_up_idx])
                if runner_confidence > 0.05:
                    visible_counter += 1
                    with st.expander(f"🔍 {questions[runner_up_idx]}"):
                        st.write(answers[runner_up_idx])
                        st.markdown(
                            f"<code style='font-size:0.75rem;'>Relevance Factor: {int(runner_confidence*100)}%</code>",
                            unsafe_allow_html=True,
                        )

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "🔍 **No primary matches cross-referenced.**\n\n"
            "We couldn't link those terms to our documentation. Try selecting an alternative macro category pill above."
        )

elif len(questions) == 0:
    st.info("⚠️ No documents exist inside this selected partition.")

# ------------------------------
# SIGNATURE ATTRIBUTION FOOTER
# ------------------------------
st.markdown(
    """
    <div class="designer-footer">
        DESIGNED AND ENGINEERED BY RAHUL UBALE
    </div>
    """,
    unsafe_allow_html=True,
)