import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()


def generate_mrm_executive_report(metrics: dict, top_features: list) -> str:
    """
    Generate an executive-level Model Risk Management (MRM) Validation Report
    following institutional financial risk advisory standards.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please configure it.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    template = """
You are a Senior Manager in Enterprise Financial Risk Advisory & Model Risk Management.
Write a formal Model Risk Management (MRM) Executive Validation Report for an Early Warning System (EWS) Credit Risk Model.

Model Specifications & Performance Metrics:
- Target Variable: 2-Year Delinquency / Probability of Default (PD)
- Model Architecture: XGBoost Classifier
- Training Set Size: {train_samples} observations
- Test Set Size: {test_samples} observations
- Model ROC-AUC Score: {roc_auc}
- Model Gini Coefficient: {gini}
- Top Risk Factors (SHAP Importance): {top_features}

Structure the output in Markdown with the following sections:
# Model Risk Management (MRM) Executive Validation Report
**Model Name:** Early Warning System (EWS) Credit Risk Model  
**Prepared By:** Senior Model Risk Management Specialist  
**Practice:** Financial Risk Advisory  

## 1. Executive Summary & Purpose
## 2. Model Performance & Validation Assessment
## 3. Key Risk Drivers Analysis (SHAP)
## 4. Model Risk & Ethical AI Considerations
## 5. Governance & Recommendations for Deployment

Do not leave empty bracket placeholders like [Your Name] or [Insert Date]. Write a complete, polished, and professional executive validation report.
"""

    prompt = PromptTemplate(
        input_variables=["train_samples", "test_samples", "roc_auc", "gini", "top_features"],
        template=template
    )

    chain = prompt | llm
    
    response = chain.invoke({
        "train_samples": f"{metrics.get('train_samples', 120000):,}",
        "test_samples": f"{metrics.get('test_samples', 30000):,}",
        "roc_auc": metrics.get('roc_auc', 0.868),
        "gini": metrics.get('gini', 0.736),
        "top_features": ", ".join(top_features)
    })

    return response.content


if __name__ == "__main__":
    sample_metrics = {
        "train_samples": 120000,
        "test_samples": 30000,
        "roc_auc": 0.868,
        "gini": 0.736
    }
    sample_features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "Age"
    ]
    
    report = generate_mrm_executive_report(sample_metrics, sample_features)
    
    os.makedirs("deliverables", exist_ok=True)
    report_path = "deliverables/sample_MRM_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"[MRM] Executive report generated and saved to: {report_path}")