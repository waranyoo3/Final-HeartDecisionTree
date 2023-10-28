import json
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Waranyoo Kunpinij Project",
    page_icon= ":bar_chart:",
)
st.sidebar.success("เลือกรายการด้านบน.")

st.write("#  🙃 การพยากรณ์การเป็นโรคหัวใจ! 👋   ")
st.write("### โดยการเอาชุดข้อมูลทั้ง8ข้อมูลมาเทรน")
st.balloons()