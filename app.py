import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

# ==========================================
# CẤU HÌNH GOOGLE SHEETS
# ==========================================
# 🛑 BẠN HÃY DÁN MÃ ID CỦA FILE GOOGLE SHEET VÀO ĐÂY:
SHEET_ID = "1I9VdzBH7j-ydr_v9Q2CXdYtenV9ethAstfYGHa5bL-E"

@st.cache_resource
def get_gspread_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    return gspread.authorize(creds)

def save_log_to_sheet(data):
    """Lưu 1 dòng dữ liệu lên Google Sheets"""
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID).sheet1
    row = [
        data['Student'], data['Class'], data['School'], data['Email'],
        data['Item_ID'], data['Skill'], data['Competency'], data['Level'],
        data['Theta_Before'], data['Correct'], data['Timestamp']
    ]
    sheet.append_row(row)

def load_logs_from_sheet():
    """Tải toàn bộ dữ liệu từ Google Sheets"""
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID).sheet1
    records = sheet.get_all_records()
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame()

# ==========================================
# THUẬT TOÁN AI (IRT & FISHER INFORMATION)
# ==========================================
def p_irt(theta, a, b):
    exponent = np.clip(-a * (theta - b), -10, 10)
    return 1 / (1 + np.exp(exponent))

def fisher_information(theta, a, b):
    p = p_irt(theta, a, b)
    return (a ** 2) * p * (1 - p)

def update_theta_mle(theta_old, history):
    if len(history) < 3:
        last_resp = history[-1]['x']
        return np.clip(theta_old + (0.5 if last_resp == 1 else -0.5), -3.0, 3.0)

    num, den = 0.0, 0.0
    for res in history:
        p = p_irt(theta_old, res['a'], res['b'])
        num += res['a'] * (res['x'] - p)
        den += (res['a'] ** 2) * p * (1 - p)
    
    if den == 0: return theta_old
    return np.clip(theta_old + (num / den), -3.0, 3.0)

# ==========================================
# KHỞI TẠO ỨNG DỤNG & PHIÊN LÀM VIỆC
# ==========================================
st.set_page_config(page_title="AI Thích Ứng - Đo lường GQVĐ & ST", layout="wide", page_icon="🧠")

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_info' not in st.session_state: st.session_state.user_info = {}
if 'theta' not in st.session_state: st.session_state.theta = 0.0
if 'history' not in st.session_state: st.session_state.history = []
if 'asked_ids' not in st.session_state: st.session_state.asked_ids = []
if 'current_item' not in st.session_state: st.session_state.current_item = None
if 'answered' not in st.session_state: st.session_state.answered = False
if 'last_correct' not in st.session_state: st.session_state.last_correct = None

@st.cache_data
def load_question_bank():
    try:
        return pd.read_csv('nganhang_lop11_chuan_hoa_1000.csv', sep=',', encoding='utf-8-sig')
    except Exception as e:
        st.error("LỖI: Không tìm thấy file ngân hàng câu hỏi.")
        st.stop()

df = load_question_bank()

# ==========================================
# GIAO DIỆN ĐĂNG NHẬP
# ==========================================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #2e6c80;'>CỔNG ĐĂNG NHẬP HỆ THỐNG AIEd</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Đo lường năng lực Giải quyết vấn đề & Sáng tạo</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            role = st.selectbox("Đăng nhập với tư cách:", ["Học sinh", "Giáo viên"])
            name = st.text_input("Họ và tên (*)", placeholder="Nhập họ và tên đầy đủ")
            cls = st.text_input("Lớp học", placeholder="Ví dụ: 11A1 (Bỏ qua nếu là Giáo viên)")
            school = st.text_input("Trường học (*)", placeholder="Trường THPT...")
            email = st.text_input("Email (*)", placeholder="Email liên hệ")
            
            submit = st.form_submit_button("🚀 Vào Hệ Thống", use_container_width=True)
            if submit:
                if name and school and email:
                    st.session_state.logged_in = True
                    st.session_state.user_info = {"role": role, "name": name, "class": cls, "school": school, "email": email}
                    st.rerun()
                else:
                    st.error("Vui lòng điền đầy đủ các thông tin bắt buộc (*)")
    st.stop()

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.markdown("<h1 style='text-align: center; color: #2e6c80;'>Hệ thống Học tập Thích ứng Lớp 11</h1>", unsafe_allow_html=True)

# SIDEBAR TỐI GIẢN
st.sidebar.markdown("### 👤 Hồ sơ đăng nhập")
st.sidebar.write(f"**Họ tên:** {st.session_state.user_info['name']}")
st.sidebar.write(f"**Vai trò:** {st.session_state.user_info['role']}")
st.sidebar.write(f"**Trường:** {st.session_state.user_info['school']}")
st.sidebar.markdown("---")
st.sidebar.info(f"📚 **Số câu đã làm: {len(st.session_state.asked_ids)}**")

if st.sidebar.button("🚪 Đăng xuất", use_container_width=True):
    st.session_state.clear()
    st.rerun()

# PHÂN QUYỀN HIỂN THỊ TABS
if st.session_state.user_info['role'] == "Học sinh":
    tabs = st.tabs(["📖 Làm bài", "📈 Tiến trình học tập"])
    tab1, tab2 = tabs[0], tabs[1]
    tab3 = None
else:
    tabs = st.tabs(["📖 Làm bài", "📈 Tiến trình học tập", "👨‍🏫 Kết quả học tập"])
    tab1, tab2, tab3 = tabs[0], tabs[1], tabs[2]

# ----------------- TAB 1: LÀM BÀI -----------------
with tab1:
    if st.session_state.current_item is None:
        available_q = df[~df['item_id'].isin(st.session_state.asked_ids)].copy()
        
        if not available_q.empty:
            DIAGNOSTIC_ITEMS = 4 # Số câu khởi động ngẫu nhiên
            
            if len(st.session_state.history) < DIAGNOSTIC_ITEMS:
                # Giai đoạn chẩn đoán: Random câu trung bình
                medium_q = available_q[(available_q['param_b'] >= -1.0) & (available_q['param_b'] <= 1.0)]
                st.session_state.current_item = medium_q.sample(1).iloc[0] if not medium_q.empty else available_q.sample(1).iloc[0]
            else:
                # Giai đoạn thích ứng: Fisher Information
                available_q['info'] = available_q.apply(
                    lambda row: fisher_information(st.session_state.theta, row['param_a'], row['param_b']), axis=1
                )
                st.session_state.current_item = available_q.sort_values(by='info', ascending=False).iloc[0]
        else:
            st.session_state.current_item = "DONE"

    q = st.session_state.current_item
    
    if isinstance(q, str) and q == "DONE":
        st.balloons()
        st.success("Tuyệt vời! Bạn đã hoàn thành toàn bộ lộ trình của hệ thống.")
    elif q is not None:
        st.markdown(f"### Câu hỏi:\n{q['question_text']}")
        
        options_dict = {'A': q['option_a'], 'B': q['option_b'], 'C': q['option_c'], 'D': q['option_d']}
        display_options = [f"A. {options_dict['A']}", f"B. {options_dict['B']}", f"C. {options_dict['C']}", f"D. {options_dict['D']}"]
        
        choice = st.radio("Lựa chọn của em:", display_options, key=f"radio_{q['item_id']}", disabled=st.session_state.answered)
        
        if not st.session_state.answered:
            if st.button("🚀 Gửi đáp án & Chẩn đoán", type="primary"):
                user_choice_letter = choice.split(".")[0]
                correct_letter = str(q['correct_answer']).strip()
                
                x = 1 if user_choice_letter == correct_letter else 0
                st.session_state.last_correct = x
                
                # Lưu lên Google Sheets
                interaction = {
                    'Student': st.session_state.user_info['name'], 'Class': st.session_state.user_info['class'],
                    'School': st.session_state.user_info['school'], 'Email': st.session_state.user_info['email'],
                    'Item_ID': q['item_id'], 'Skill': q['skill_id'],
                    'Competency': q['competency'], 'Level': q['level'],
                    'Theta_Before': round(st.session_state.theta, 3),
                    'Correct': x, 'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                save_log_to_sheet(interaction)
                
                st.session_state.asked_ids.append(q['item_id'])
                st.session_state.history.append({'a': q['param_a'], 'b': q['param_b'], 'x': x})
                st.session_state.theta = update_theta_mle(st.session_state.theta, st.session_state.history)
                st.session_state.answered = True
                st.rerun()

        if st.session_state.answered:
            if st.session_state.last_correct == 1:
                st.success("🎉 Chính xác! Em làm rất tốt.")
            else:
                st.error("❌ Rất tiếc, đáp án chưa chính xác. Em hãy đọc kỹ gợi ý bên dưới để hiểu rõ bản chất bài toán nhé!")
                st.warning(f"💡 **GỢI Ý:**\n- **Bước 1:** {q['scaffold_step1']}\n- **Bước 2:** {q['scaffold_step2']}")
            
            if st.button("👉 Tiếp tục sang câu tiếp theo", type="primary"):
                st.session_state.answered = False
                st.session_state.current_item = None
                st.rerun()

# ----------------- TAB 2: TIẾN TRÌNH HỌC TẬP -----------------
with tab2:
    st.markdown("### 📊 Biểu đồ Năng lực & Chẩn đoán Lỗ hổng")
    all_logs = load_logs_from_sheet()
    
    if not all_logs.empty:
        my_logs = all_logs[all_logs['Student'] == st.session_state.user_info['name']]
        if not my_logs.empty:
            col1, col2 = st.columns(2)
            with col1:
                my_logs['Question_Num'] = range(1, len(my_logs) + 1)
                fig_theta = px.line(my_logs, x='Question_Num', y='Theta_Before', 
                                    title='Sự phát triển Năng lực theo thời gian',
                                    markers=True, text='Correct',
                                    labels={'Theta_Before': 'Mức Năng lực', 'Question_Num': 'Thứ tự câu hỏi'})
                fig_theta.update_traces(textposition="bottom right")
                fig_theta.update_xaxes(dtick=1) 
                st.plotly_chart(fig_theta, use_container_width=True)
                
            with col2:
                skill_data = my_logs.groupby('Skill')['Correct'].mean() * 100
                skill_df = skill_data.reset_index()
                skill_df.columns = ['Mã Kỹ Năng', 'Tỷ lệ đúng (%)']
                fig_skill = px.bar(skill_df, x='Mã Kỹ Năng', y='Tỷ lệ đúng (%)', 
                                   title='Mức độ làm chủ Kỹ năng',
                                   color='Tỷ lệ đúng (%)', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_skill, use_container_width=True)
                
            st.markdown("#### Đánh giá Năng lực Cốt lõi (GDPT 2018)")
            comp_data = my_logs.groupby('Competency').agg(
                Tổng_Câu=('Item_ID', 'count'), Số_Câu_Đúng=('Correct', 'sum')
            )
            comp_data['Độ Thành Thạo (%)'] = round((comp_data['Số_Câu_Đúng'] / comp_data['Tổng_Câu']) * 100, 1)
            st.dataframe(comp_data, use_container_width=True)
        else:
            st.info("Em hãy hoàn thành một số câu hỏi để hệ thống có thể vẽ biểu đồ năng lực nhé!")
    else:
        st.info("Hệ thống chưa ghi nhận dữ liệu.")

# ----------------- TAB 3: QUẢN LÝ KẾT QUẢ (CHỈ GIÁO VIÊN) -----------------
if tab3 is not None:
    with tab3:
        st.markdown("### 👨‍🏫 Bảng Điều Khiển Dành Cho Giáo Viên")
        all_logs = load_logs_from_sheet()
        
        if not all_logs.empty:
            st.markdown("#### 1. Tổng quan Lớp học")
            summary_df = all_logs.groupby(['Student', 'Class', 'School']).agg(
                Số_câu_đã_làm=('Item_ID', 'count'),
                Số_câu_đúng=('Correct', 'sum'),
                Năng_lực_hiện_tại=('Theta_Before', 'last')
            ).reset_index()
            
            summary_df['Tỷ lệ đúng (%)'] = round((summary_df['Số_câu_đúng'] / summary_df['Số_câu_đã_làm']) * 100, 1)
            summary_df['Năng_lực_hiện_tại'] = round(summary_df['Năng_lực_hiện_tại'], 2)
            st.dataframe(summary_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 2. Phân tích chi tiết theo Học sinh")
            student_list = summary_df['Student'].tolist()
            selected_student = st.selectbox("Chọn một học sinh để phân tích lộ trình học:", student_list)
            
            if selected_student:
                student_details = all_logs[all_logs['Student'] == selected_student].sort_values(by='Timestamp', ascending=False)
                st.write(f"Nhật ký làm bài của học sinh **{selected_student}**:")
                st.dataframe(student_details, use_container_width=True)
        else:
            st.info("Chưa có học sinh nào nộp bài lên hệ thống.")
