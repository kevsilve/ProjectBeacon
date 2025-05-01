import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openai
from openai import OpenAI

from utils.models import grad_rate, srs_from_literacy, juvenile_risk, college_readiness
from utils.gpt_prompt import generate_prompt

# --- CONFIG ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Updated for OpenAI v1

# --- COUNTY DATA ---
data = {
    "Lawrence": {
        "Absenteeism": 30.8,
        "Literacy": 88,
        "Youth_Programs": 4,
        "Internet_Access": 83.8,
        "Goals": {"SRS": 35.0, "Grad": 90.0, "Readiness": 45.0}
    },
    "Monroe": {
        "Absenteeism": 18.4,
        "Literacy": 92,
        "Youth_Programs": 8,
        "Internet_Access": 91.2,
        "Goals": {"SRS": 30.0, "Grad": 92.0, "Readiness": 50.0}
    }
}

# --- STREAMLIT APP ---
st.title("📍 Project Beacon Simulator")
st.write("Use local education, health, and infrastructure data to predict youth outcomes and plan strategic interventions.")

county = st.selectbox("Select a County", list(data.keys()))
goal_type = st.selectbox("Set a Goal", ["Reduce SRS < X", "Increase Graduation Rate > X", "Boost College Readiness > X"])

baseline = data[county]
goals = baseline.get("Goals", {"SRS": 35.0, "Grad": 90.0, "Readiness": 45.0})
with st.expander("📊 Current Baseline Metrics"):
    st.json(baseline)

st.subheader("🎛️ Adjust Metrics Manually")
adjusted_absenteeism = st.slider("Absenteeism (%)", 0.0, 50.0, float(baseline["Absenteeism"]), 0.1)
adjusted_literacy = st.slider("Literacy Proficiency (%)", 60.0, 100.0, float(baseline["Literacy"]), 0.1)
adjusted_programs = st.slider("Youth Programs (count)", 0, 15, int(baseline["Youth_Programs"]))
adjusted_internet = st.slider("Homes with Internet (%)", 50.0, 100.0, float(baseline["Internet_Access"]), 0.1)

# Projections
srs = srs_from_literacy(adjusted_literacy)
grad = grad_rate(adjusted_absenteeism)
readiness = college_readiness(adjusted_internet)
juv_risk = juvenile_risk(adjusted_programs)

# Define goal thresholds and deltas
target_srs = goals["SRS"]
target_grad = goals["Grad"]
target_readiness = goals["Readiness"]
srs_delta = srs - target_srs
grad_delta = grad - target_grad
readiness_delta = readiness - target_readiness

# Color-coded formatting
def delta_color(value):
    if value > 0:
        return "🔴"
    elif value < 0:
        return "🟢"
    else:
        return "⚪"

st.subheader("📈 Projected Outcomes (With Adjustments)")
st.metric("Graduation Rate", f"{grad:.1f}%", delta=f"{grad_delta:+.1f} from goal {delta_color(-grad_delta)}")
st.metric("Success Risk Score (SRS)", f"{srs:.1f}", delta=f"{srs_delta:+.1f} from goal {delta_color(-srs_delta)}")
st.metric("College Readiness Index", f"{readiness:.1f}", delta=f"{readiness_delta:+.1f} from goal {delta_color(-readiness_delta)}")
st.metric("Juvenile Risk Index", f"{juv_risk:.1f}")

# --- GPT-BASED ACTION PLAN ---
st.subheader("🧠 Suggested Intervention Plan")
adjusted_data = {
    "Absenteeism": adjusted_absenteeism,
    "Literacy": adjusted_literacy,
    "Youth_Programs": adjusted_programs,
    "Internet_Access": adjusted_internet
}
prompt = generate_prompt(county, adjusted_data, goal_type)

if st.button("💡 Generate Action Plan"):
    with st.spinner("Thinking like a policy analyst..."):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a community strategist and education planner."},
                {"role": "user", "content": prompt}
            ]
        )
        st.write(response.choices[0].message.content)

fig, ax = plt.subplots(figsize=(8, 4))
metrics = ["Grad Rate", "SRS", "Readiness", "Juvenile Risk"]
values = [grad, srs, readiness, juv_risk]
colors = ["seagreen", "crimson", "orange", "dodgerblue"]
ax.bar(metrics, values, color=colors)
ax.set_title(f"{county} County – Adjusted Outcomes")
st.pyplot(fig)

st.caption("Built with ❤️ using GPT + Streamlit")
