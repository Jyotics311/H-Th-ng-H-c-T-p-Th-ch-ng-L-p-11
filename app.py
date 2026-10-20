import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

# ==========================================
# 1. CẤU HÌNH GOOGLE SHEETS
# ==========================================
SHEET_ID = "1I9VdzBH7j-ydr_v9Q2CXdYtenV9ethAstfYGHa5bL-E"

@st.cache_resource
def get_gspread_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    return gspread.authorize(creds)

def save_log_to_sheet(data):
    try:
        client = get_gspread_client()
        sheet = client.open_by_key(SHEET_ID).get_worksheet(0)
        row = [
            data['Student'], data['Class'], data['School'], data['Email'],
            data['Item_ID'], data['Skill'], data['Competency'], data['Level'],
            data['Theta_Before'], data['Correct'], data['Timestamp']
        ]
        sheet.append_row(row)
    except Exception as e:
        st.toast(f"Cảnh báo: Không thể lưu kết nối - {e}")

def load_logs_from_sheet():
    try:
        client = get_gspread_client()
        sheet = client.open_by_key(SHEET_ID).get_worksheet(0)
        data = sheet.get_all_values()
        
        if len(data) > 1:
            headers = data[0]
            records = data[1:]
            df = pd.DataFrame(records, columns=headers)
            df = df.loc[:, ~df.columns.duplicated()]
            if "" in df.columns:
                df = df.drop(columns=[""])
            return df
        else:
            return pd.DataFrame() 
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# 2. THUẬT TOÁN AI (IRT & FISHER)
# ==========================================
def p_irt(theta, a, b):
    return 1 / (1 + np.exp(np.clip(-a * (theta - b), -10, 10)))

def fisher_information(theta, a, b):
    p = p_irt(theta, a, b)
    return (a ** 2) * p * (1 - p)

def update_theta_mle(theta_old, history):
    if len(history) < 4:
        return np.clip(theta_old + (0.5 if history[-1]['x'] == 1 else -0.5), -3.0, 3.0)

    num, den = 0.0, 0.0
    for res in history:
        p = p_irt(theta_old, res['a'], res['b'])
        num += res['a'] * (res['x'] - p)
        den += (res['a'] ** 2) * p * (1 - p)
    
    return np.clip(theta_old + (num / den), -3.0, 3.0) if den > 0 else theta_old

# ==========================================
# 3. KHỞI TẠO ỨNG DỤNG
# ==========================================
st.set_page_config(page_title="AI Thích Ứng - Lớp 11", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    html, body, [class*="st-"] {
        font-size: 18px !important;
    }
    button[data-baseweb="tab"] div {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    .stMarkdown p {
        font-size: 19px !important;
        line-height: 1.6 !important;
    }
    div[data-testid="stRadio"] > label > div > p {
        font-size: 19px !important;
        font-weight: bold !important;
    }
    div[role="radiogroup"] label p {
        font-size: 19px !important;
    }
    div[role="radiogroup"] {
        gap: 15px !important; 
    }
    </style>
""", unsafe_allow_html=True)

for key in ['logged_in', 'answered']:
    if key not in st.session_state: st.session_state[key] = False
for key in ['user_info', 'current_item', 'last_correct', 'last_skill']:
    if key not in st.session_state: st.session_state[key] = None
for key in ['history', 'asked_ids', 'recent_skills']:
    if key not in st.session_state: st.session_state[key] = []
if 'theta' not in st.session_state: st.session_state.theta = 0.0

@st.cache_data
def load_question_bank():
    file_name = 'nganhang_lop11_chuan_hoa.csv'
    try:
        df = pd.read_csv(file_name, sep=',', encoding='utf-8-sig', on_bad_lines='skip')
        if len(df.columns) < 5: 
            df = pd.read_csv(file_name, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
        
        if 'param_a' in df.columns and 'param_b' in df.columns:
            df['param_a'] = pd.to_numeric(df['param_a'].astype(str).str.replace(',', '.'), errors='coerce')
            df['param_b'] = pd.to_numeric(df['param_b'].astype(str).str.replace(',', '.'), errors='coerce')
            df = df.dropna(subset=['param_a', 'param_b'])
        return df
    except Exception as e:
        st.error(f"Lỗi đọc Ngân hàng câu hỏi: {e}")
        return pd.DataFrame()

df = load_question_bank()

# ==========================================
# 4. GIAO DIỆN ĐĂNG NHẬP
# ==========================================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #2e6c80;'>CỔNG ĐĂNG NHẬP HỆ THỐNG AIEd</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            role = st.selectbox("Đăng nhập với tư cách:", ["Học sinh", "Giáo viên"])
            name = st.text_input("Họ và tên (*)")
            cls = st.text_input("Lớp học", placeholder="Bắt buộc với Học sinh")
            school = st.text_input("Trường học (*)")
            email = st.text_input("Email (*)")
            
            if st.form_submit_button("🚀 Vào Hệ Thống", use_container_width=True):
                if role == "Học sinh" and not cls:
                    st.error("Học sinh vui lòng nhập Lớp học!")
                elif name and school and email:
                    st.session_state.logged_in = True
                    st.session_state.user_info = {"role": role, "name": name, "class": cls if cls else "", "school": school, "email": email}
                    st.rerun()
                else:
                    st.error("Vui lòng điền đầy đủ các thông tin (*)")
    st.stop()

# ==========================================
# 5. GIAO DIỆN CHÍNH
# ==========================================
st.markdown("<h1 style='text-align: center; color: #2e6c80;'>Hệ thống Học tập Thích ứng Lớp 11</h1>", unsafe_allow_html=True)

st.sidebar.markdown("### 👤 Hồ sơ đăng nhập")
st.sidebar.write(f"**Họ tên:** {st.session_state.user_info['name']}")
st.sidebar.write(f"**Vai trò:** {st.session_state.user_info['role']}")
st.sidebar.write(f"**Trường:** {st.session_state.user_info['school']}")
st.sidebar.info(f"📚 **Số câu đã làm: {len(st.session_state.asked_ids)}**")

if st.sidebar.button("🚪 Đăng xuất", use_container_width=True):
    st.session_state.clear()
    st.rerun()

tabs = st.tabs(["📖 Làm bài", "📈 Tiến trình học tập"]) if st.session_state.user_info['role'] == "Học sinh" else st.tabs(["📖 Làm bài", "📈 Tiến trình học tập", "👨‍🏫 Kết quả học tập"])
tab1, tab2 = tabs[0], tabs[1]
tab3 = tabs[2] if len(tabs) == 3 else None

# ----------------- TAB 1: LÀM BÀI -----------------
with tab1:
    MAX_QUESTIONS = 15
    
    if len(st.session_state.asked_ids) >= MAX_QUESTIONS:
        st.success(f"Em đã hoàn thành bài Test ({MAX_QUESTIONS} câu).")
        correct_count = sum([res['x'] for res in st.session_state.history])
        st.info(f"📊 **Kết quả:** Em đã làm đúng **{correct_count}/{MAX_QUESTIONS}** câu.")
        st.write("Em hãy sang tab **Tiến trình học tập** để xem biểu đồ chi tiết kết quả của mình nhé!")
        
        if st.button("🔄 Bắt đầu bài Test mới", type="primary"):
            st.session_state.asked_ids = []
            st.session_state.history = []
            st.session_state.recent_skills = []
            st.session_state.current_item = None
            st.session_state.answered = False
            st.session_state.last_skill = None
            st.session_state.theta = 0.0
            st.rerun()
            
    else:
        st.markdown(f"**Tiến độ bài Test:** {len(st.session_state.asked_ids) + 1} / {MAX_QUESTIONS}")
        st.progress(len(st.session_state.asked_ids) / MAX_QUESTIONS)
        
        if st.session_state.current_item is None:
            available_q = df[~df['item_id'].isin(st.session_state.asked_ids)].copy()
            
            if not available_q.empty:
                DIAGNOSTIC_ITEMS = 4 
                
                if len(st.session_state.history) < DIAGNOSTIC_ITEMS:
                    medium_q = available_q[(available_q['param_b'] >= -1.0) & (available_q['param_b'] <= 1.0)]
                    st.session_state.current_item = medium_q.sample(1).iloc[0] if not medium_q.empty else available_q.sample(1).iloc[0]
                else:
                    available_q['distance'] = (available_q['param_b'] - st.session_state.theta).abs()
                    closest_q = available_q.nsmallest(20, 'distance').copy()
                    closest_q['info'] = closest_q.apply(
                        lambda r: fisher_information(st.session_state.theta, r['param_a'], r['param_b']), axis=1
                    )
                    st.session_state.current_item = closest_q.nlargest(5, 'info').sample(1).iloc[0]
            else:
                st.session_state.current_item = "DONE"

        q = st.session_state.current_item
        
        if isinstance(q, str) and q == "DONE":
            st.success("Hệ thống đã hết câu hỏi phù hợp cho em trong ngân hàng.")
        elif q is not None:
            st.markdown(f"### Câu hỏi:\n{q['question_text']}")
            
            options_dict = {'A': q['option_a'], 'B': q['option_b'], 'C': q['option_c'], 'D': q['option_d']}
            display_options = [f"A. {options_dict['A']}", f"B. {options_dict['B']}", f"C. {options_dict['C']}", f"D. {options_dict['D']}"]
            
            choice = st.radio("Lựa chọn của em:", display_options, key=f"radio_{q['item_id']}", disabled=st.session_state.answered)
            
            if not st.session_state.answered:
                if st.button("Gửi đáp án", type="primary"):
                    user_choice_letter = choice.split(".")[0]
                    correct_letter = str(q['correct_answer']).strip()
                    
                    x = 1 if user_choice_letter == correct_letter else 0
                    
                    st.session_state.last_correct = x
                    st.session_state.last_skill = q['skill_id'] 
                    
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
                    st.success("🎉 Chính xác!")
                else:
                    st.error("❌ Rất tiếc, đáp án chưa chính xác. Em hãy đọc kỹ gợi ý từng bước bên dưới nhé!")
                    st.warning(f"💡 **GỢI Ý KHẮC PHỤC:**\n- **Bước 1:** {q['scaffold_step1']}\n- **Bước 2:** {q['scaffold_step2']}")
                    
                    with st.expander("👀 Bấm vào đây để xem đáp án đúng "):
                        st.markdown(f"**Đáp án chính xác là: {q['correct_answer']}**")
                
                if st.button("👉 Tiếp tục sang câu tiếp theo", type="primary"):
                    st.session_state.answered = False
                    st.session_state.current_item = None
                    st.rerun()

# ----------------- TAB 2: TIẾN TRÌNH HỌC TẬP -----------------
with tab2:
    st.markdown("### 📊 Quỹ đạo phát triển Năng lực cá nhân")
    all_logs = load_logs_from_sheet()
    
    if not all_logs.empty:
        my_logs = all_logs[all_logs['Student'] == st.session_state.user_info['name']].copy()
        
        if not my_logs.empty:
            my_logs['Theta_Before'] = pd.to_numeric(my_logs['Theta_Before'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
            my_logs['Correct'] = pd.to_numeric(my_logs['Correct'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
            my_logs['Timestamp'] = pd.to_datetime(my_logs['Timestamp'], errors='coerce')
            
            my_logs = my_logs.sort_values('Timestamp')
            MAX_Q = 15
            my_logs['Test_Attempt'] = (np.arange(len(my_logs)) // MAX_Q) + 1
            
            test_summary = my_logs.groupby('Test_Attempt').agg(
                Thời_gian=('Timestamp', 'last'),
                Số_câu_đúng=('Correct', 'sum'),
                Tổng_câu=('Item_ID', 'count')
            ).reset_index()
            
            test_summary['Tỷ lệ đúng (%)'] = round((test_summary['Số_câu_đúng'] / test_summary['Tổng_câu']) * 100, 1)
            test_summary['Text_Hiển_thị'] = test_summary['Số_câu_đúng'].astype(str) + "/" + test_summary['Tổng_câu'].astype(str) + " câu"

            fig_score = px.line(test_summary, x='Thời_gian', y='Tỷ lệ đúng (%)', 
                                title='Sự tiến bộ của em qua các bài Test',
                                markers=True, text='Text_Hiển_thị',
                                labels={'Thời_gian': 'Thời gian nộp bài', 'Tỷ lệ đúng (%)': 'Tỷ lệ làm đúng (%)'})
            
            fig_score.update_traces(textposition="top center", line=dict(width=3), marker=dict(size=10))
            fig_score.update_xaxes(tickformat="%d/%m\n%H:%M") 
            fig_score.update_yaxes(range=[-5, 110])
            st.plotly_chart(fig_score, use_container_width=True)
                
            st.markdown("#### Đánh giá Năng lực Cốt lõi (GDPT 2018)")
            comp_data = my_logs.groupby('Competency').agg(Tổng_Câu=('Item_ID', 'count'), Số_Câu_Đúng=('Correct', 'sum'))
            comp_data['Độ Thành Thạo (%)'] = round((comp_data['Số_Câu_Đúng'] / comp_data['Tổng_Câu']) * 100, 1)
            st.dataframe(comp_data, use_container_width=True)
        else:
            st.info("Em hãy hoàn thành một số câu hỏi để hệ thống có thể vẽ biểu đồ năng lực nhé!")
    else:
        st.info("Hệ thống chưa ghi nhận dữ liệu.")

# ----------------- TAB 3: QUẢN LÝ KẾT QUẢ -----------------
if tab3 is not None:
    with tab3:
        teacher_school = str(st.session_state.user_info['school']).strip().title()
        
        st.markdown(f"### 👨‍🏫 Bảng Điều Khiển: Trường {teacher_school}")
        all_logs = load_logs_from_sheet()
        
        if not all_logs.empty:
            all_logs['Theta_Before'] = pd.to_numeric(all_logs['Theta_Before'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
            all_logs['Correct'] = pd.to_numeric(all_logs['Correct'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
            
            all_logs['Student'] = all_logs['Student'].astype(str).str.strip().str.title()
            all_logs['School'] = all_logs['School'].astype(str).str.strip().str.title()
            all_logs['Class'] = all_logs['Class'].astype(str).str.strip().str.upper()
            
            school_logs = all_logs[all_logs['School'] == teacher_school].copy()
            if not school_logs.empty:
                available_classes = sorted([c for c in school_logs['Class'].unique() if str(c).strip() != ""])
                if available_classes:
                    selected_class = st.selectbox("📚 Chọn lớp học:", available_classes)
                    class_logs = school_logs[school_logs['Class'] == selected_class].copy()
                    
                    st.markdown(f"#### 1. Tổng quan Lớp {selected_class}")
                    summary_df = class_logs.groupby(['Student']).agg(
                        Số_câu_đã_làm=('Item_ID', 'count'), 
                        Số_câu_đúng=('Correct', 'sum'),
                        Năng_lực_hiện_tại=('Theta_Before', 'last')
                    ).reset_index()
                    
                    summary_df['Tỷ lệ đúng (%)'] = round((summary_df['Số_câu_đúng'] / summary_df['Số_câu_đã_làm']) * 100, 1)
                    summary_df['Năng_lực_hiện_tại'] = round(summary_df['Năng_lực_hiện_tại'], 2)
                    
                    summary_df = summary_df.rename(columns={
                        'Student': 'Họ và tên',
                        'Số_câu_đã_làm': 'Số câu đã làm',
                        'Số_câu_đúng': 'Số câu đúng',
                        'Năng_lực_hiện_tại': 'Năng lực (Theta) hiện tại'
                    })
                    
                    st.dataframe(summary_df, use_container_width=True)
                    
                    st.markdown("#### 2. Phân tích chi tiết Học sinh")
                    selected_student = st.selectbox("👤 Chọn học sinh:", summary_df['Họ và tên'].tolist())
                    
                    if selected_student:
                        my_logs_teacher = class_logs[class_logs['Student'] == selected_student].copy()
                        my_logs_teacher['Timestamp'] = pd.to_datetime(my_logs_teacher['Timestamp'], errors='coerce')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            my_logs_teacher = my_logs_teacher.sort_values('Timestamp')
                            MAX_Q = 15
                            my_logs_teacher['Test_Attempt'] = (np.arange(len(my_logs_teacher)) // MAX_Q) + 1
                            
                            test_summary_t = my_logs_teacher.groupby('Test_Attempt').agg(
                                Thời_gian=('Timestamp', 'last'),
                                Số_câu_đúng=('Correct', 'sum'),
                                Tổng_câu=('Item_ID', 'count')
                            ).reset_index()
                            
                            test_summary_t['Tỷ lệ đúng (%)'] = round((test_summary_t['Số_câu_đúng'] / test_summary_t['Tổng_câu']) * 100, 1)
                            test_summary_t['Text_Hiển_thị'] = test_summary_t['Số_câu_đúng'].astype(str) + "/" + test_summary_t['Tổng_câu'].astype(str) + " câu"

                            fig_score_t = px.line(test_summary_t, x='Thời_gian', y='Tỷ lệ đúng (%)', 
                                                title=f'Sự tiến bộ của {selected_student}',
                                                markers=True, text='Text_Hiển_thị',
                                                labels={'Thời_gian': 'Thời gian', 'Tỷ lệ đúng (%)': 'Tỷ lệ đúng (%)'})
                            fig_score_t.update_traces(textposition="top center", line=dict(width=3), marker=dict(size=10))
                            fig_score_t.update_xaxes(tickformat="%d/%m\n%H:%M") 
                            fig_score_t.update_yaxes(range=[-5, 110])
                            st.plotly_chart(fig_score_t, use_container_width=True)
                            
                        with col2:
                            skill_data_t = my_logs_teacher.groupby('Skill')['Correct'].mean() * 100
                            skill_df_t = skill_data_t.reset_index()
                            skill_df_t.columns = ['Mã Kỹ Năng', 'Tỷ lệ đúng (%)']
                            fig_skill_t = px.bar(skill_df_t, x='Mã Kỹ Năng', y='Tỷ lệ đúng (%)', 
                                               title='Mức độ làm chủ Kỹ năng',
                                               color='Tỷ lệ đúng (%)', color_continuous_scale='RdYlGn')
                            st.plotly_chart(fig_skill_t, use_container_width=True)
                        
                        with st.expander("🔎 Xem chi tiết quá trình:"):
                            st.dataframe(my_logs_teacher.sort_values(by='Timestamp', ascending=False), use_container_width=True)
                else:
                    st.warning("Chưa có dữ liệu phân lớp.")
            else:
                st.warning("Trường học này chưa có dữ liệu học sinh.")
