import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

SHEET_ID = "1I9VdzBH7j-ydr_v9Q2CXdYtenV9ethAstfYGHa5bL-E"

@st.cache_resource
def get_gspread_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    return gspread.authorize(creds)

def save_log_to_sheet(data):
    try:
        sheet = get_gspread_client().open_by_key(SHEET_ID).get_worksheet(0)
        sheet.append_row([data[k] for k in ['Student','Class','School','Email','Item_ID','Skill','Competency','Level','Theta_Before','Correct','Timestamp']])
    except Exception as e: st.toast(f"Lỗi lưu kết nối: {e}")

def load_logs_from_sheet():
    try:
        data = get_gspread_client().open_by_key(SHEET_ID).get_worksheet(0).get_all_values()
        if len(data) > 1:
            df = pd.DataFrame(data[1:], columns=data[0])
            df = df.loc[:, ~df.columns.duplicated()]
            return df.drop(columns=[""]) if "" in df.columns else df
        return pd.DataFrame() 
    except: return pd.DataFrame()

# Data Cleaning Helper
def clean_num(series, fill_val=0.0):
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce').fillna(fill_val)

# IRT & MLE Algorithms
def p_irt(theta, a, b): return 1 / (1 + np.exp(np.clip(-a * (theta - b), -10, 10)))
def fisher_information(theta, a, b): p = p_irt(theta, a, b); return (a ** 2) * p * (1 - p)

def update_theta_mle(theta_old, history):
    if len(history) < 3: return np.clip(theta_old + (0.5 if history[-1]['x'] == 1 else -0.5), -3.0, 3.0)
    num, den = 0.0, 0.0
    for res in history:
        p = p_irt(theta_old, res['a'], res['b'])
        num += res['a'] * (res['x'] - p)
        den += (res['a'] ** 2) * p * (1 - p)
    return np.clip(theta_old + (num / den), -3.0, 3.0) if den > 0 else theta_old

# Application Setup
st.set_page_config(page_title="AI Thích Ứng - Lớp 11", layout="wide", page_icon="🧠")
st.markdown("""<style>html, body, [class*="st-"] {font-size: 18px !important;} button[data-baseweb="tab"] div {font-size: 20px !important; font-weight: bold !important;} .stMarkdown p, div[data-testid="stRadio"] > label > div > p, div[role="radiogroup"] label p {font-size: 19px !important;} div[role="radiogroup"] {gap: 15px !important;}</style>""", unsafe_allow_html=True)

for k in ['logged_in', 'answered']: st.session_state.setdefault(k, False)
for k in ['user_info', 'current_item', 'last_correct', 'last_skill']: st.session_state.setdefault(k, None)
for k in ['history', 'asked_ids', 'recent_skills']: st.session_state.setdefault(k, [])
st.session_state.setdefault('theta', 0.0)

@st.cache_data
def load_question_bank():
    try:
        df = pd.read_csv('nganhang_lop11_chuan_hoa.csv', sep=',', encoding='utf-8-sig', on_bad_lines='skip')
        if len(df.columns) < 5: df = pd.read_csv('nganhang_lop11_chuan_hoa.csv', sep=';', encoding='utf-8-sig', on_bad_lines='skip')
        if 'param_a' in df.columns and 'param_b' in df.columns:
            df['param_a'] = clean_num(df['param_a'])
            df['param_b'] = clean_num(df['param_b'])
            df = df.dropna(subset=['param_a', 'param_b'])
        return df
    except: return pd.DataFrame()

df = load_question_bank()

# Auth System
if not st.session_state.logged_in:
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>CỔNG ĐĂNG NHẬP AIEd</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            role = st.selectbox("Vai trò:", ["Học sinh", "Giáo viên"])
            name, cls, school, email = st.text_input("Họ tên (*)"), st.text_input("Lớp"), st.text_input("Trường (*)"), st.text_input("Email (*)")
            if st.form_submit_button("Vào Hệ Thống", use_container_width=True):
                if role == "Học sinh" and not cls: st.error("Vui lòng nhập Lớp!")
                elif name and school and email:
                    st.session_state.update({'logged_in': True, 'user_info': {"role": role, "name": name, "class": cls, "school": school, "email": email}})
                    st.rerun()
                else: st.error("Điền đầy đủ thông tin (*)")
    st.stop()

# Main UI
st.markdown("<h1 style='text-align: center; color: #2e6c80;'>Hệ thống Học tập Thích ứng Lớp 11</h1>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Họ tên:** {st.session_state.user_info['name']}\n\n**Trường:** {st.session_state.user_info['school']}")
if st.sidebar.button("Đăng xuất", use_container_width=True): st.session_state.clear(); st.rerun()

tabs = st.tabs(["Làm bài", "Tiến trình học tập"] if st.session_state.user_info['role'] == "Học sinh" else ["Làm bài", "Tiến trình", "Kết quả học tập"])

# TAB 1: Adaptive Testing
with tabs[0]:
    MAX_Q = 15
    if len(st.session_state.asked_ids) >= MAX_Q:
        st.success(f"Hoàn thành bài Test. Kết quả: {sum(r['x'] for r in st.session_state.history)}/{MAX_Q} câu.")
        if st.button("Bắt đầu Test mới", type="primary"):
            st.session_state.update({'asked_ids': [], 'history': [], 'current_item': None, 'answered': False, 'theta': 0.0})
            st.rerun()
    else:
        st.progress(len(st.session_state.asked_ids) / MAX_Q, text=f"Tiến độ: {len(st.session_state.asked_ids)+1}/{MAX_Q}")
        if st.session_state.current_item is None:
            available_q = df[~df['item_id'].isin(st.session_state.asked_ids)].copy()
            if not available_q.empty:
                if len(st.session_state.history) < 4: # Diagnostic Phase
                    mq = available_q[(available_q['param_b'] >= -1.0) & (available_q['param_b'] <= 1.0)]
                    st.session_state.current_item = mq.sample(1).iloc[0] if not mq.empty else available_q.sample(1).iloc[0]
                else: # Adaptive Phase (Proximity + Fisher)
                    available_q['dist'] = (available_q['param_b'] - st.session_state.theta).abs()
                    closest_q = available_q.nsmallest(20, 'dist').copy()
                    closest_q['info'] = closest_q.apply(lambda r: fisher_information(st.session_state.theta, r['param_a'], r['param_b']), axis=1)
                    st.session_state.current_item = closest_q.nlargest(5, 'info').sample(1).iloc[0]
            else: st.session_state.current_item = "DONE"

        q = st.session_state.current_item
        if q == "DONE": st.success("Hết câu hỏi phù hợp.")
        elif q is not None:
            st.markdown(f"### {q['question_text']}")
            opts = [f"A. {q['option_a']}", f"B. {q['option_b']}", f"C. {q['option_c']}", f"D. {q['option_d']}"]
            choice = st.radio("Lựa chọn:", opts, disabled=st.session_state.answered)
            
            if not st.session_state.answered and st.button("Gửi đáp án", type="primary"):
                x = 1 if choice.split(".")[0] == str(q['correct_answer']).strip() else 0
                st.session_state.last_correct = x
                save_log_to_sheet({'Student': st.session_state.user_info['name'], 'Class': st.session_state.user_info['class'], 'School': st.session_state.user_info['school'], 'Email': st.session_state.user_info['email'], 'Item_ID': q['item_id'], 'Skill': q['skill_id'], 'Competency': q['competency'], 'Level': q['level'], 'Theta_Before': round(st.session_state.theta, 3), 'Correct': x, 'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S")})
                st.session_state.asked_ids.append(q['item_id'])
                st.session_state.history.append({'a': q['param_a'], 'b': q['param_b'], 'x': x})
                st.session_state.theta = update_theta_mle(st.session_state.theta, st.session_state.history)
                st.session_state.answered = True
                st.rerun()

            if st.session_state.answered:
                if st.session_state.last_correct == 1: st.success("Chính xác!")
                else:
                    st.error("Chưa chính xác. Hãy xem gợi ý:")
                    st.warning(f"- Bước 1: {q['scaffold_step1']}\n- Bước 2: {q['scaffold_step2']}")
                    with st.expander("Xem đáp án đúng"): st.markdown(f"**{q['correct_answer']}**")
                if st.button("Câu tiếp theo", type="primary"):
                    st.session_state.update({'answered': False, 'current_item': None}); st.rerun()

# TAB 2: Student Dashboard
with tabs[1]:
    all_logs = load_logs_from_sheet()
    if not all_logs.empty:
        ml = all_logs[all_logs['Student'] == st.session_state.user_info['name']].copy()
        if not ml.empty:
            ml['Correct'], ml['Timestamp'] = clean_num(ml['Correct']), pd.to_datetime(ml['Timestamp'], errors='coerce')
            ml['Test_Attempt'] = (np.arange(len(ml)) // 15) + 1
            ts = ml.groupby('Test_Attempt').agg(Time=('Timestamp','last'), Score=('Correct','sum'), Total=('Item_ID','count')).reset_index()
            ts['Rate'] = round((ts['Score'] / ts['Total']) * 100, 1)
            ts['Label'] = ts['Score'].astype(str) + "/" + ts['Total'].astype(str)
            
            fig = px.line(ts, x='Time', y='Rate', title='Tiến bộ qua các bài Test', markers=True, text='Label', labels={'Time':'Thời gian', 'Rate':'Tỷ lệ đúng (%)'})
            fig.update_traces(textposition="top center", line=dict(width=3), marker=dict(size=10))
            fig.update_xaxes(tickformat="%d/%m\n%H:%M"); fig.update_yaxes(range=[-5, 110])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Năng lực Cốt lõi (GDPT 2018)")
            comp = ml.groupby('Competency').agg(Tổng_Câu=('Item_ID','count'), Số_Câu_Đúng=('Correct','sum'))
            comp['Độ Thành Thạo (%)'] = round((comp['Số_Câu_Đúng'] / comp['Tổng_Câu']) * 100, 1)
            st.dataframe(comp, use_container_width=True)

# TAB 3: Teacher Dashboard
if len(tabs) == 3:
    with tabs[2]:
        tschool = str(st.session_state.user_info['school']).strip().title()
        st.markdown(f"### Bảng Điều Khiển: Trường {tschool}")
        if not all_logs.empty:
            al = all_logs.copy()
            al['Theta_Before'], al['Correct'] = clean_num(al['Theta_Before']), clean_num(al['Correct'])
            al['Student'], al['School'], al['Class'] = al['Student'].astype(str).str.strip().str.title(), al['School'].astype(str).str.strip().str.title(), al['Class'].astype(str).str.strip().str.upper()
            sl = al[al['School'] == tschool].copy()
            
            if not sl.empty:
                classes = sorted([c for c in sl['Class'].unique() if str(c).strip() != ""])
                if classes:
                    sel_class = st.selectbox("Chọn lớp:", classes)
                    cl = sl[sl['Class'] == sel_class].copy()
                    
                    st.markdown("#### 1. Tổng quan Lớp")
                    sdf = cl.groupby('Student').agg(Làm=('Item_ID','count'), Đúng=('Correct','sum'), Theta=('Theta_Before','last')).reset_index()
                    sdf['Tỷ lệ (%)'], sdf['Theta'] = round((sdf['Đúng']/sdf['Làm'])*100, 1), round(sdf['Theta'], 2)
                    st.dataframe(sdf.rename(columns={'Student':'Họ tên', 'Làm':'Số câu', 'Đúng':'Đúng', 'Theta':'Năng lực'}), use_container_width=True)
                    
                    st.markdown("#### 2. Chi tiết Học sinh")
                    sel_stu = st.selectbox("Học sinh:", sdf['Student'].tolist())
                    if sel_stu:
                        mlt = cl[cl['Student'] == sel_stu].copy()
                        mlt['Timestamp'] = pd.to_datetime(mlt['Timestamp'], errors='coerce')
                        c1, c2 = st.columns(2)
                        
                        with c1:
                            mlt['Test_Attempt'] = (np.arange(len(mlt)) // 15) + 1
                            tst = mlt.groupby('Test_Attempt').agg(Time=('Timestamp','last'), Score=('Correct','sum'), Total=('Item_ID','count')).reset_index()
                            tst['Rate'], tst['Label'] = round((tst['Score']/tst['Total'])*100, 1), tst['Score'].astype(str) + "/" + tst['Total'].astype(str)
                            f1 = px.line(tst, x='Time', y='Rate', title=f'Tiến bộ: {sel_stu}', markers=True, text='Label')
                            f1.update_traces(textposition="top center", line=dict(width=3), marker=dict(size=10))
                            f1.update_xaxes(tickformat="%d/%m\n%H:%M"); f1.update_yaxes(range=[-5, 110])
                            st.plotly_chart(f1, use_container_width=True)
                        
                        with c2:
                            sk = mlt.groupby('Skill')['Correct'].mean() * 100
                            f2 = px.bar(x=sk.index, y=sk.values, title='Làm chủ Kỹ năng', color=sk.values, color_continuous_scale='RdYlGn')
                            st.plotly_chart(f2, use_container_width=True)
                        
                        with st.expander("Xem chi tiết dữ liệu (Raw Data)"): st.dataframe(mlt.sort_values('Timestamp', ascending=False), use_container_width=True)
                else:
                    st.warning("Chưa có dữ liệu phân lớp.")
            else:
                st.warning("Trường học này chưa có dữ liệu học sinh.")
