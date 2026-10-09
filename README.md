# 🛡️ GenAI Risk Copilot — Enterprise Early Warning System (EWS) & MRM Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)](https://www.langchain.com/)
[![OpenAI GPT-4o-mini](https://img.shields.io/badge/OpenAI-GPT--4o--mini-black.svg)](https://openai.com/)

# 🛡️ GenAI Risk Copilot — Enterprise EWS & MRM Platform

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ews-genai-risk-copilot-n8zaqoyt4rdmrf3vsrvf3m.streamlit.app)
[![Live Interactive Demo](https://img.shields.io/badge/Demo-Live%20App-blue?style=flat&logo=streamlit)](https://ews-genai-risk-copilot-n8zaqoyt4rdmrf3vsrvf3m.streamlit.app)

An end-to-end, enterprise-grade **GenAI Risk Copilot** built for Financial Risk Advisory, Model Risk Management (MRM), and Early Warning System (EWS) monitoring in banking and credit portfolios.

---

## 🌟 Key Architecture & Capabilities

1. **📊 Predictive ML & Interpretability Engine**
   - **XGBoost Classifier** predicting 2-Year Delinquency / Probability of Default (PD) on 150,000 observations.
   - **Model Metrics:** ROC-AUC `0.868`, Gini Coefficient `0.736`.
   - **SHAP Explainability:** Horizontal interactive visualizations breaking down top risk factors (e.g., Revolving Line Utilization, Past Due Frequency, Debt Ratio).

2. **📝 Automated MRM Executive Report Generator**
   - LLM pipeline powered by **LangChain** and **GPT-4o-mini**.
   - Generates formal Model Risk Management validation reports adhering to supervisory standards (e.g., Federal Reserve SR 11-7).
   - One-click exportable `.md` validation documentation.

3. **💬 Regulatory Policy RAG Copilot**
   - RAG engine built with **ChromaDB**, **OpenAI Embeddings (`text-embedding-3-small`)**, and **LCEL chains**.
   - Ingests policy frameworks and regulatory papers into vector stores to deliver grounded, non-hallucinating policy context.

4. **🎯 Real-Time Borrower EWS Credit Scoring Simulator**
   - Interactive credit committee simulator allowing risk officers to adjust borrower indicators and evaluate real-time PD scores and risk tiers (🟢 Normal / 🟡 Amber Watchlist / 🔴 Critical Alert).

---

## 📁 Repository Structure

```text
EWS-GenAI-Risk-Copilot/
├── data/
│   ├── docs/                   # PDF policy papers & regulatory guidelines
│   └── raw/                    # Credit risk dataset
├── deliverables/               # Generated MRM markdown reports
├── src/
│   ├── credit_model.py         # XGBoost model training & evaluation
│   ├── data_pipeline.py        # Feature engineering & preprocessing
│   ├── mrm_generator.py       # Automated MRM validation report generator
│   └── rag_engine.py           # ChromaDB vectorstore & LCEL RAG pipeline
├── app.py                      # Main Interactive Streamlit Application
├── .env                        # Local environment variables (API keys)
├── .gitignore                  # Git exclusions file
└── requirements.txt            # Dependency configuration
```
