import streamlit as st
import sqlite3
import datetime
from sentence_transformers import SentenceTransformer, util
import torch  # needed for tensors
import pandas as pd

# --------------------------
# Setup embedding model
# --------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("ibm-granite/granite-embedding-small-english-r2")

embed_model = load_embed_model()

# Pre-defined example messages for intent detection
intent_examples = {
    "add income": [
        "I earned 5000",
        "My salary is 10000",
        "I got paid today",
        "Received payment"
    ],
    "add expense": [
        "I spent 200 on food",
        "Bought groceries for 500",
        "Paid rent 3000",
        "I gave 100 to my friend"
    ],
    "show summary": [
        "What is my balance?",
        "Show my summary",
        "How much left",
        "What are my expenses?"
    ]
}

# Precompute embeddings for examples
example_embeddings = {}
for intent, texts in intent_examples.items():
    example_embeddings[intent] = embed_model.encode(texts, convert_to_tensor=True)

def classify_intent(msg: str):
    msg_emb = embed_model.encode(msg, convert_to_tensor=True)
    scores = {}
    for intent, ex_embs in example_embeddings.items():
        sims = util.cos_sim(msg_emb, ex_embs)
        scores[intent] = float(torch.max(sims).cpu().numpy())
    best_intent = max(scores, key=lambda x: scores[x])
    return best_intent, scores[best_intent]

# --------------------------
# DB + finance functions
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
        (t_type, amount, category, description, datetime.date.today().isoformat())
    )
    conn.commit()
    conn.close()

def edit_transaction(t_id, t_type, amount, category, description):
    conn = sqlite3.connect("finance.db")
    c = conn.cursor()
    c.execute(
        "UPDATE transactions SET type=?, amount=?, category=?, description=? WHERE id=?",
        (t_type, amount, category, description, t_id)
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
# Streamlit app
# --------------------------
st.set_page_config(page_title="💸 Personal Finance Chatbot (Granite)", layout="wide")
st.title("💸 Personal Finance Bot (using IBM Granite embeddings)")

tab1, tab2 = st.tabs(["Chat & Manual Entry", "Transaction Records"])

init_db()

with tab1:
    st.subheader("Chat with Finance Bot")
    user_input = st.text_input("💬 Type your message:")

    if st.button("Submit Chat"):
        if user_input.strip():
            intent, score = classify_intent(user_input)
            st.write(f"Detected intent: **{intent}** (similarity {score:.3f})")

            if intent == "add income":
                amount = [float(s) for s in user_input.split() if s.replace('.','',1).isdigit()]
                if amount:
                    add_transaction("Income", amount[0], "General", user_input)
                    st.success(f"✅ Added income: {amount[0]}")
                else:
                    st.warning("No numerical amount detected in message.")

            elif intent == "add expense":
                amount = [float(s) for s in user_input.split() if s.replace('.','',1).isdigit()]
                category = "General"
                if "for" in user_input.lower():
                    category = user_input.lower().split("for")[-1].strip().split()[0]
                if amount:
                    add_transaction("Expense", amount[0], category, user_input)
                    st.error(f"💸 Added expense: {category} - {amount[0]}")
                else:
                    st.warning("No numerical amount detected in message.")

            elif intent == "show summary":
                income, expenses, remaining = get_summary()
                st.info(f"💰 Income: {income} | 📉 Expenses: {expenses} | 📊 Remaining: {remaining}")

            else:
                st.write("🤖 I didn't understand intent. Try simpler phrasing.")

    st.subheader("Manual Entry")
    with st.form("manual"):
        etype = st.selectbox("Type", ["Income", "Expense"])
        amt = st.number_input("Amount", min_value=0.0, step=100.0)
        category = st.text_input("Category", "")
        description = st.text_input("Description (optional)", "")
        submitted = st.form_submit_button("Add")

        if submitted:
            add_transaction(etype, amt, category or "General", description or "")
            st.success(f"{etype} of {amt} added.")

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
                new_type = st.selectbox("Type", ["Income", "Expense"], index=0 if t_type=="Income" else 1)
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
