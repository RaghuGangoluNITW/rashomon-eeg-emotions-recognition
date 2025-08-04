# SHAP and other interpretability methods.
import shap

def explain_with_shap(model, data):
    explainer = shap.Explainer(model, data)
    shap_values = explainer(data)
    shap.summary_plot(shap_values, data)
