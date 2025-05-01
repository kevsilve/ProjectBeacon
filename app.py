import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openai

from utils.models import grad_rate, srs_from_literacy, juvenile_risk, college_readiness
from utils.gpt_prompt import generate_prompt

# --- CONFIG ---
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Add your OpenAI API key to Streamlit secrets

# --- COUNTY DATA ---
data = {
    "Lawrence": {
        "Absenteeism": 30.8,
        "Literacy": 88,
        "Youth_Programs": 4,
        "Internet_Access": 83.8,
    },
    "Monroe": {
        "Absenteeism": 18.4,
        "Literacy": 92,
        "Youth_Programs": 8,
        "Internet_Access": 91.2,
    }
}

# --- STREAMLIT APP ---
st.title("📍 Project Beacon Simulator")
st.write("Use local education, health, and infrastructure data to predict youth outcomes and plan strategic interventions.")

county = st.selectbox("Select a County", list(data.keys()))
goal_type = st.selectbox("Set a Goal", ["Reduce SRS < 35", "Increase Graduation Rate > 90%", "Boost College Readiness > 45"])

baseline = data[county]
with st.expander("📊 Current Baseline Metrics"):
    st.json(baseline)

# Projections
srs = srs_from_literacy(baseline["Literacy"])
grad = grad_rate(baseline["Absenteeism"])
readiness = college_readiness(baseline["Internet_Access"])
juv_risk = juvenile_risk(baseline["Youth_Programs"])

st.subheader("📈 Projected Outcomes (Current)")
st.metric("Graduation Rate", f"{grad:.1f}%")
st.metric("Success Risk Score (SRS)", f"{srs:.1f}")
st.metric("College Readiness Index", f"{readiness:.1f}")
st.metric("Juvenile Risk Index", f"{juv_risk:.1f}")

# --- GPT-BASED ACTION PLAN ---
st.subheader("🧠 Suggested Intervention Plan")
prompt = generate_prompt(county, baseline, goal_type)

if st.button("💡 Generate Action Plan"):
    with st.spinner("Thinking like a policy analyst..."):
        response = openai.ChatCompletion.create(
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
ax.set_title(f"{county} County – Baseline Outcomes")
st.pyplot(fig)

st.caption("Built with ❤️ using GPT + Streamlit")
