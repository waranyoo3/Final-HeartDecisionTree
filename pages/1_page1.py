import json
import time
import requests
import streamlit as st
import tkinter as tk

root = tk.Tk()
background_image = tk.PhotoImage(file="background.png")
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

st.set_page_config(
    page_title="Waranyoo Kunpinij Project",
    page_icon= ":bar_chart:",
)
st.sidebar.success("เลือกรายการด้านล่าง.")

st.write("#🙃การพยากรณ์การเป็นโรคหัวใจ!👋")
st.write("##โดยการเอาชุดข้อมูลทั้ง8ข้อมูลมาเทรน")
st.write("ขนาดของชุดข้อมูลคือ 1,319 ตัวอย่าง ซึ่งมีเก้าช่อง โดยที่ 8 ช่องเป็นช่องอินพุต และ 1 ช่องสำหรับช่องเอาท์พุต อายุ เพศ(0 สำหรับผู้หญิง, 1 สำหรับผู้ชาย (แรงกระตุ้น), ความดันโลหิตซิสโตลิก (ความดันสูง), ความดันโลหิตล่าง (ความดันต่ำ), น้ำตาลในเลือด (กลูโคส), CK-MB (kcm) และทดสอบ-โทรโปนิน (โทรโปนิน ) เป็นตัวแทนของฟิลด์อินพุต ในขณะที่ฟิลด์เอาต์พุตเกี่ยวข้องกับภาวะหัวใจวาย (คลาส) ซึ่งแบ่งออกเป็นสองประเภท (เชิงลบและบวก) ค่าลบหมายถึงการไม่มีอาการหัวใจวาย ในขณะที่ค่าบวกหมายถึงการไม่มีอาการหัวใจวาย) , อัตราการเต้นของหัวใจ ")
st.balloons()