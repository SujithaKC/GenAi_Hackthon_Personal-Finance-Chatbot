import streamlit as st
import sqlite3
import datetime
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import subprocess

# --------------------------
# Load Granite Embedding Model
# --------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("ibm-granite/granite-embedding-small-english-r2")

embed_model = load_embed_model()

# --------------------------
# Ollama Granite Local Model
# --------------------------
def get_granite_answer(prompt: str, system_message: str = None) -> str:
    if system_message is None:
        system_message = "You are a helpful, concise financial advisor. Provide practical advice about personal finance, budgeting, saving, and investing."

    full_prompt = f"System: {system_message}\nUser: {prompt}\nAssistant:"

    try:
        result = subprocess.run(
            ["ollama", "run", "granite3.1-moe:3b", full_prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"❌ Ollama Error: {result.stderr.strip()}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

# --------------------------
# DB Functions
# --------------------------
def init_db():
    conn = sqlite3.connect("finance.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            amount REAL,
            category TEXT,
            description TEXT,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_transaction(t_type, amount, category="General", description=""):
    conn = sqlite3.connect("finance.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO transactions (type, amount, category, description, date) VALUES (?, ?, ?, ?, ?)",
        (t_type, amount, category, description, datetime.date.today().isoformat()),
    )
    conn.commit()
    conn.close()

def edit_transaction(t_id, t_type, amount, category, description):
    conn = sqlite3.connect("finance.db")
    c = conn.cursor()
    c.execute(
        "UPDATE transactions SET type=?, amount=?, category=?, description=? WHERE id=?",
        (t_type, amount, category, description, t_id),
    )
    conn.commit()
    conn.close()

def delete_transaction(t_id):
    conn = sqlite3.connect("finance.db")
    c = conn.cursor()
    c.execute("DELETE FROM transactions WHERE id=?", (t_id,))
    conn.commit()
    conn.close()

def get_summary():
    conn = sqlite3.connect("finance.db")
    c = conn.cursor()
    c.execute("SELECT SUM(amount) FROM transactions WHERE type='Income'")
    income = c.fetchone()[0] or 0
    c.execute("SELECT SUM(amount) FROM transactions WHERE type='Expense'")
    expenses = c.fetchone()[0] or 0
    conn.close()
    return income, expenses, income - expenses

def get_transactions():
    conn = sqlite3.connect("finance.db")
    c = conn.cursor()
    c.execute("SELECT * FROM transactions ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# --------------------------
# Personalized Financial Advice
# --------------------------
def get_personalized_saving_tips():
    income, expenses, remaining = get_summary()
    savings_rate = (remaining / income * 100) if income > 0 else 0
    prompt = f"""
    User's Financial Summary:
    - Monthly Income: ₹{income}
    - Monthly Expenses: ₹{expenses}
    - Monthly Savings: ₹{remaining}
    - Savings Rate: {savings_rate:.1f}%

    Provide 3-4 specific, actionable saving tips tailored to this user's situation.
    """
    system_message = "You are an expert financial advisor specializing in personal savings strategies."
    return get_granite_answer(prompt, system_message)

def get_personalized_investment_tips():
    income, expenses, remaining = get_summary()
    transactions = get_transactions()
    expense_categories = {}
    for t in transactions:
        if t[1] == "Expense":
            cat = t[3]
            amt = t[2]
            expense_categories[cat] = expense_categories.get(cat, 0) + amt
    top_expenses = sorted(expense_categories.items(), key=lambda x: x[1], reverse=True)[:3]
    prompt = f"""
    User's Financial Profile:
    - Monthly Income: ₹{income}
    - Monthly Expenses: ₹{expenses}
    - Monthly Investable Amount: ₹{remaining}
    - Top Expense Categories: {', '.join([f'{c} (₹{a})' for c,a in top_expenses]) if top_expenses else 'No data'}

    Provide personalized investment advice considering income, spending, and investable funds.
    """
    system_message = "You are a certified investment advisor. Provide practical, personalized investment recommendations."
    return get_granite_answer(prompt, system_message)

def get_financial_health_analysis():
    income, expenses, remaining = get_summary()
    savings_rate = (remaining / income * 100) if income > 0 else 0
    expense_ratio = (expenses / income * 100) if income > 0 else 100
    prompt = f"""
    User's Financial Health Metrics:
    - Monthly Income: ₹{income}
    - Monthly Expenses: ₹{expenses}
    - Monthly Savings: ₹{remaining}
    - Savings Rate: {savings_rate:.1f}%
    - Expense-to-Income Ratio: {expense_ratio:.1f}%

    Provide a comprehensive financial health analysis with actionable insights.
    """
    system_message = "You are a financial health analyst. Provide thorough, actionable insights."
    return get_granite_answer(prompt, system_message)

# --------------------------
# Intent Detection
# --------------------------
intent_examples = {
    "add income": ["I earned 5000", "My salary is 10000", "I got paid today"],
    "add expense": ["I spent 200 on food", "Bought groceries for 500", "Paid rent 3000"],
    "show summary": ["What is my balance?", "Show my summary", "How much left"],
    "saving tips": ["give me saving tips", "how can I save more?", "help me save money"],
    "investment tips": ["investment advice", "where should I invest?", "help me invest"],
    "financial analysis": ["analyze my finances", "financial health check", "how am I doing financially?"],
}

example_embeddings = {
    intent: embed_model.encode(texts, convert_to_tensor=True)
    for intent, texts in intent_examples.items()
}

def classify_intent(msg: str):
    msg_emb = embed_model.encode(msg, convert_to_tensor=True)
    scores = {}
    for intent, ex_embs in example_embeddings.items():
        sims = util.cos_sim(msg_emb, ex_embs)
        scores[intent] = float(torch.max(sims).cpu().numpy())
    best_intent = max(scores, key=lambda x: scores[x])
    return best_intent, scores[best_intent]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="💸 Personal Finance Chatbot", layout="wide")
st.title("💸 Personal Finance Bot (Granite Local + Embeddings)")

init_db()

tab1, tab2, tab3 = st.tabs(["Chat & Manual Entry", "Transaction Records", "Financial Insights"])

with tab1:
    st.subheader("Chat with Finance Bot")
    user_input = st.text_input("💬 Type your message:")

    if st.button("Submit Chat"):
        if user_input.strip():
            intent, score = classify_intent(user_input)
            if score > 0.6:
                if intent == "add income":
                    amt = [float(s) for s in user_input.split() if s.replace(".", "", 1).isdigit()]
                    if amt:
                        add_transaction("Income", amt[0], "General", user_input)
                        st.success(f"✅ Added income: {amt[0]}")
                elif intent == "add expense":
                    amt = [float(s) for s in user_input.split() if s.replace(".", "", 1).isdigit()]
                    cat = "General"
                    if "for" in user_input.lower():
                        cat = user_input.lower().split("for")[-1].strip().split()[0]
                    if amt:
                        add_transaction("Expense", amt[0], cat, user_input)
                        st.error(f"💸 Added expense: {cat} - {amt[0]}")
                elif intent == "show summary":
                    income, expenses, remaining = get_summary()
                    st.info(f"💰 Income: {income} | 📉 Expenses: {expenses} | 📊 Remaining: {remaining}")
                elif intent == "saving tips":
                    with st.spinner("Generating personalized saving tips..."):
                        tips = get_personalized_saving_tips()
                    st.success("💡 **Personalized Saving Tips:**")
                    st.write(tips)
                elif intent == "investment tips":
                    with st.spinner("Generating investment recommendations..."):
                        advice = get_personalized_investment_tips()
                    st.success("📈 **Personalized Investment Advice:**")
                    st.write(advice)
                elif intent == "financial analysis":
                    with st.spinner("Analyzing your financial health..."):
                        analysis = get_financial_health_analysis()
                    st.success("🔍 **Financial Health Analysis:**")
                    st.write(analysis)
            else:
                with st.spinner("🤖 Thinking with Granite..."):
                    response = get_granite_answer(user_input)
                    st.write(response)

    st.subheader("Manual Entry")
    with st.form("manual"):
        etype = st.selectbox("Type", ["Income", "Expense"])
        amt = st.number_input("Amount", min_value=0.0, step=100.0)
        category = st.text_input("Category", "")
        description = st.text_input("Description (optional)", "")
        submitted = st.form_submit_button("Add")

        if submitted:
            add_transaction(etype, amt, category or "General", description or "")
            st.success(f"{etype} of ₹{amt} added.")

with tab2:
    st.subheader("Transaction Records")
    income, expenses, remaining = get_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"₹{income}")
    col2.metric("Total Expenses", f"₹{expenses}")
    col3.metric("Net Balance", f"₹{remaining}")

    rows = get_transactions()
    if rows:
        df = pd.DataFrame(rows, columns=["ID", "Type", "Amount", "Category", "Description", "Date"])
        st.dataframe(df, use_container_width=True)

        st.subheader("✏️ Edit / 🗑️ Delete Transaction")
        ids = [r[0] for r in rows]
        selected_id = st.selectbox("Select Transaction ID", ids)

        if selected_id:
            record = [r for r in rows if r[0] == selected_id][0]
            t_id, t_type, amount, category, description, date = record

            with st.form("edit_delete_form"):
                new_type = st.selectbox("Type", ["Income", "Expense"], index=0 if t_type == "Income" else 1)
                new_amount = st.number_input("Amount", value=amount)
                new_category = st.text_input("Category", value=category)
                new_desc = st.text_input("Description", value=description)

                col1, col2 = st.columns(2)
                with col1:
                    save = st.form_submit_button("💾 Save Changes")
                with col2:
                    delete = st.form_submit_button("🗑️ Delete")

                if save:
                    edit_transaction(t_id, new_type, new_amount, new_category, new_desc)
                    st.success("Transaction updated. Refresh to see changes.")
                if delete:
                    delete_transaction(t_id)
                    st.warning("Transaction deleted. Refresh to see changes.")
    else:
        st.info("No transactions yet.")

with tab3:
    st.subheader("📊 Financial Insights Dashboard")
    income, expenses, remaining = get_summary()

    if income > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Savings Rate", f"{(remaining/income*100):.1f}%")
        col2.metric("Expense Ratio", f"{(expenses/income*100):.1f}%")
        col3.metric("Monthly Savings", f"₹{remaining}")

        if st.button("🔄 Refresh Financial Insights", key="refresh_insights"):
            with st.spinner("Generating comprehensive financial insights..."):
                tips = get_personalized_saving_tips()
                advice = get_personalized_investment_tips()
                analysis = get_financial_health_analysis()

                st.success("💡 **Personalized Saving Strategies:**")
                st.write(tips)
                st.divider()

                st.success("📈 **Investment Recommendations:**")
                st.write(advice)
                st.divider()

                st.success("🔍 **Comprehensive Financial Health Analysis:**")
                st.write(analysis)
    else:
        st.warning("Add some income and expense data to get personalized financial insights!")
