import glob
import os
import re
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# ==================== 1. CẤU HÌNH GIAO DIỆN HỆ THỐNG ====================
st.set_page_config(
    page_title="FIFA Multi-Year AI Analytics & Valuation Pipeline",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ HỆ THỐNG PHÂN TÍCH & ĐỊNH GIÁ CẦU THỦ THEO CHUỖI THỜI GIAN")
st.caption(
    "Đồ án Khoa học Dữ liệu & Big Data Pipeline | Tích hợp Machine Learning"
    " Valuation Engine"
)

# Hiển thị 6 mốc Pipeline trực quan
col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)
col_p1.info("1️⃣ **Multi-Sources**\n\n`Archives (4-10)`")
col_p2.info("2️⃣ **Auto Ingestion**\n\n`Smart CSV Scan`")
col_p3.info("3️⃣ **ETL Process**\n\n`Panel Data Clean`")
col_p4.info("4️⃣ **Storage**\n\n`Streamlit Cache`")
col_p5.info("5️⃣ **ML / AI**\n\n`Linear Regression`")
col_p6.success("6️⃣ **Visualization**\n\n`Dynamic Time-Series`")

st.divider()


# ==================== 2. TẦNG INGESTION, ETL & MACHINE LEARNING AI ENGINE ====================
@st.cache_data
def load_and_train_ai_model():
  all_csv_files = []

  # Quét tất cả các thư mục archive từ (4) đến (10) trong Downloads
  for i in range(4, 11):
    folder_path = f"C:/Users/ADMIN/Downloads/archive ({i})"
    if os.path.exists(folder_path):
      found_files = glob.glob(
          os.path.join(folder_path, "**/*.csv"), recursive=True
      )
      all_csv_files.extend(found_files)

  if not all_csv_files:
    return pd.DataFrame()

  columns_to_keep = [
      "sofifa_id",
      "short_name",
      "age",
      "player_positions",
      "overall",
      "potential",
      "value_eur",
      "wage_eur",
      "club_name",
      "nationality_name",
      "nationality",
      "pace",
      "shooting",
      "passing",
      "dribbling",
      "defending",
      "physic",
      "fifa_version",
  ]

  data_frames = []
  for file_path in all_csv_files:
    file_lower = file_path.lower()
    # Chỉ quét các file chứa thông tin cầu thủ chuẩn từ bộ dữ liệu
    if any(
        kw in file_lower
        for kw in ["player", "male", "football", "data", "fifa"]
    ):
      try:
        temp_df = pd.read_csv(file_path, usecols=lambda c: c in columns_to_keep)

        if {"short_name", "overall", "value_eur"}.issubset(temp_df.columns):
          # Nhận diện năm phiên bản từ tên file hoặc thư mục
          version_val = 2024
          match = re.search(r"(20\d{2}|\b\d{2}\b)", file_path)
          if match:
            val = match.group(0)
            if len(val) == 2:
              version_val = int("20" + val)
            else:
              version_val = int(val)
          else:
            if "4" in file_path:
              version_val = 2020
            elif "5" in file_path:
              version_val = 2021
            elif "6" in file_path:
              version_val = 2022
            elif "7" in file_path:
              version_val = 2023
            elif "8" in file_path:
              version_val = 2024
            elif "9" in file_path or "10" in file_path:
              version_val = 2025

          if (
              "fifa_version" not in temp_df.columns
              or temp_df["fifa_version"].isnull().all()
          ):
            temp_df["fifa_version"] = version_val
          else:
            temp_df["fifa_version"] = temp_df["fifa_version"].fillna(
                version_val
            )

          # Chuẩn hóa năm về dạng 4 chữ số chuẩn
          temp_df["fifa_version"] = temp_df["fifa_version"].apply(
              lambda x: int(2000 + x) if x < 100 else int(x)
          )

          # Đồng bộ tên cột quốc tịch
          if (
              "nationality" in temp_df.columns
              and "nationality_name" not in temp_df.columns
          ):
            temp_df = temp_df.rename(columns={"nationality": "nationality_name"})

          data_frames.append(temp_df)
      except Exception:
        pass

  if not data_frames:
    return pd.DataFrame()

  master_df = pd.concat(data_frames, ignore_index=True)
  master_df = master_df.dropna(
      subset=["short_name", "value_eur", "overall", "potential"]
  ).copy()

  # Xử lý giá trị thiếu
  master_df["nationality_name"] = master_df["nationality_name"].fillna(
      "Không rõ quốc tịch"
  )
  master_df["club_name"] = master_df["club_name"].fillna("Cầu thủ tự do")
  master_df["player_positions"] = master_df["player_positions"].fillna("Chưa rõ")

  feature_cols = [
      "age",
      "overall",
      "potential",
      "pace",
      "shooting",
      "passing",
      "dribbling",
      "defending",
      "physic",
  ]
  for col in feature_cols:
    if col not in master_df.columns:
      master_df[col] = 50
    else:
      master_df[col] = master_df[col].fillna(master_df[col].median())

  # Huấn luyện mô hình Machine Learning (Linear Regression)
  X = master_df[feature_cols]
  y = master_df["value_eur"]

  ml_model = LinearRegression()
  ml_model.fit(X, y)

  master_df["AI_Predicted_Value"] = ml_model.predict(X).clip(min=200000)

  return master_df


# Mốc 4: Storage
df = load_and_train_ai_model()

if df.empty:
  st.error(
      "❌ Không tìm thấy file dữ liệu cầu thủ hợp lệ trong các thư mục"
      " archive! Vui lòng kiểm tra lại đường dẫn."
  )
  st.stop()


# ==================== 3. THANH ĐIỀU HƯỚNG (SIDEBAR) ====================
st.sidebar.title("🎛️ Bảng Điều Hướng")
menu_selection = st.sidebar.radio(
    "Chọn phân hệ chức năng:",
    [
        "🔍 Tra cứu & Quỹ đạo sự nghiệp",
        "⚖️ So sánh đối đầu (Head-to-Head)",
        "🏆 Top cầu thủ giá trị nhất",
        "📈 Phân tích chuỗi thời gian (EDA)",
        "⚙️ Trạng thái Pipeline & Dữ liệu",
    ],
)


# ==================== 4. MODULE 1: TRA CỨU & QUỸ ĐẠO SỰ NGHIỆP ====================
if menu_selection == "🔍 Tra cứu & Quỹ đạo sự nghiệp":
  st.sidebar.header("🔍 Lọc Cầu Thủ & Năm")
  player_list = sorted(df["short_name"].dropna().unique())
  selected_player = st.sidebar.selectbox("Chọn tên cầu thủ:", player_list)

  player_history = df[df["short_name"] == selected_player].sort_values(
      "fifa_version"
  )

  if not player_history.empty:
    available_versions = sorted(player_history["fifa_version"].unique())
    selected_version = st.sidebar.selectbox(
        "📅 Chọn năm phiên bản:",
        available_versions,
        index=len(available_versions) - 1,
        format_func=lambda x: f"Năm {x} (FIFA {str(x)[-2:]})",
    )

    p = player_history[
        player_history["fifa_version"] == selected_version
    ].iloc[0]

    st.header(
        f"👤 Hồ Sơ Cầu Thủ: {p['short_name'].upper()} (Năm {selected_version})"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Tuổi năm {selected_version}", f"{int(p['age'])} tuổi")
    c2.metric("Vị trí thi đấu", p["player_positions"])
    c3.metric("Quốc tịch", p["nationality_name"])
    c4.metric("Chỉ số Overall", int(p["overall"]))
    c5.metric("Tiềm năng Potential", int(p["potential"]))

    st.write(
        f"🏢 **Câu lạc bộ năm {selected_version}:**"
        f" `{p.get('club_name', 'Không rõ')}`"
    )
    st.divider()

    st.subheader(
        f"🤖 Kết Quả AI Định Giá (Năm {selected_version} - Machine Learning)"
    )
    val_actual = p["value_eur"]
    val_pred = p["AI_Predicted_Value"]
    diff = val_pred - val_actual

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        f"💵 Giá Thị Trường Gốc ({selected_version})", f"{val_actual:,.0f} EUR"
    )
    col_b.metric("🤖 AI Machine Learning Dự Đoán", f"{val_pred:,.0f} EUR")

    if diff > 2000000:
      col_c.success(
          f"💡 Nhận định: **ĐỊNH GIÁ THẤP** ({diff:,.0f} EUR) -> 🟢 **NÊN MUA!**"
      )
    elif diff < -2000000:
      col_c.error(
          f"⚠️ Nhận định: **ĐỊNH GIÁ CAO** ({abs(diff):,.0f} EUR) -> 🔴"
          " **CÂN NHẮC!**"
      )
    else:
      col_c.info("🎯 Nhận định: **ĐÚNG GIÁ THỊ TRƯỜNG** -> 🟡 **ỔN ĐỊNH!**")

    st.divider()

    st.subheader(
        "📈 Quỹ Đạo Phát Triển Sự Nghiệp & Biến Động Giá Trị Qua Các Năm"
    )
    if len(player_history) > 1:
      chart_col1, chart_col2 = st.columns(2)
      with chart_col1:
        st.write("**Biến động Overall & Potential qua các năm**")
        st.line_chart(
            player_history.set_index("fifa_version")[["overall", "potential"]]
        )
      with chart_col2:
        st.write("**Biến động Giá trị chuyển nhượng (EUR) qua các năm**")
        st.line_chart(
            player_history.set_index("fifa_version")[["value_eur"]]
        )
    else:
      st.info(
          "💡 Cầu thủ này hiện chỉ có dữ liệu ở một phiên bản năm trong hệ"
          " thống."
      )

    st.subheader(
        f"📊 Biểu Đồ Kỹ Năng Chi Tiết (Năm {selected_version})"
    )
    skills_df = pd.DataFrame({
        "Kỹ năng": [
            "Tốc độ",
            "Sút",
            "Chuyền",
            "Rê bóng",
            "Phòng thủ",
            "Thể lực",
        ],
        "Điểm số": [
            p.get("pace", 50),
            p.get("shooting", 50),
            p.get("passing", 50),
            p.get("dribbling", 50),
            p.get("defending", 50),
            p.get("physic", 50),
        ],
    })
    st.bar_chart(skills_df.set_index("Kỹ năng"), color="#1f77b4")


# ==================== 5. MODULE 2: SO SÁNH HEAD-TO-HEAD ====================
elif menu_selection == "⚖️ So sánh đối đầu (Head-to-Head)":
  st.subheader("⚖️ So Sánh Trực Tiếp Hai Cầu Thủ Theo Năm")
  player_list = sorted(df["short_name"].dropna().unique())

  col_p1, col_p2 = st.columns(2)
  with col_p1:
    p1_name = st.selectbox("Chọn cầu thủ số 1:", player_list, index=0)
  with col_p2:
    p2_name = st.selectbox(
        "Chọn cầu thủ số 2:",
        player_list,
        index=1 if len(player_list) > 1 else 0,
    )

  p1 = df[df["short_name"] == p1_name].iloc[-1]
  p2 = df[df["short_name"] == p2_name].iloc[-1]

  comparison_table = pd.DataFrame(
      {
          f"{p1['short_name']} (Năm {int(p1['fifa_version'])})": [
              int(p1["age"]),
              int(p1["overall"]),
              int(p1["potential"]),
              p1.get("pace", 0),
              p1.get("shooting", 0),
              p1.get("passing", 0),
              f"{p1['value_eur']:,.0f} EUR",
          ],
          f"{p2['short_name']} (Năm {int(p2['fifa_version'])})": [
              int(p2["age"]),
              int(p2["overall"]),
              int(p2["potential"]),
              p2.get("pace", 0),
              p2.get("shooting", 0),
              p2.get("passing", 0),
              f"{p2['value_eur']:,.0f} EUR",
          ],
      },
      index=[
          "Tuổi",
          "Overall",
          "Potential",
          "Tốc độ",
          "Sút",
          "Chuyền",
          "Giá trị thị trường",
      ],
  )
  st.dataframe(comparison_table, width="stretch")


# ==================== 6. MODULE 3: TOP XẾP HẠNG ====================
elif menu_selection == "🏆 Top cầu thủ giá trị nhất":
  st.subheader("🏆 Bảng Xếp Hạng Cầu Thủ Có Giá Trị Cao Nhất Toàn Hệ Thống")
  top_n_slider = st.slider("Tùy chỉnh số lượng hiển thị:", 5, 30, 10)

  top_df = df.nlargest(top_n_slider, "value_eur")[[
      "short_name",
      "club_name",
      "age",
      "overall",
      "fifa_version",
      "value_eur",
      "AI_Predicted_Value",
  ]]
  st.dataframe(top_df.reset_index(drop=True), width="stretch")


# ==================== 7. MODULE 4: PHÂN TÍCH EDA CHUYÊN SÂU ====================
elif menu_selection == "📈 Phân tích chuỗi thời gian (EDA)":
  st.subheader(
      "📈 Khám Phá Dữ Liệu Lớn & Chuỗi Thời Gian (Panel Data Exploration)"
  )
  st.write(
      "Phân tích mối quan hệ tương quan và phân phối dữ liệu trên toàn bộ"
      " Master Dataset đa niên độ."
  )

  eda_c1, eda_c2 = st.columns(2)
  with eda_c1:
    st.write("### 📉 Tương quan: Overall vs Giá trị thị trường")
    st.scatter_chart(
        df.sample(min(2000, len(df))), x="overall", y="value_eur", color="age"
    )
    st.caption("Biểu đồ phân tán thể hiện mức tăng trưởng giá trị theo Overall.")

  with eda_c2:
    st.write("### 📊 Thống kê bản ghi theo Phiên bản năm")
    version_counts = df["fifa_version"].value_counts().sort_index()
    st.bar_chart(version_counts)
    st.caption("Số lượng cầu thủ đóng góp từ các nguồn archive.")


# ==================== 8. MODULE 5: CẤU HÌNH PIPELINE ====================
elif menu_selection == "⚙️ Trạng thái Pipeline & Dữ liệu":
  st.subheader("⚙️ Thông Tin Kỹ Thuật Hệ Thống & Trạng Thái Pipeline")
  st.write(
      "📂 **Nguồn dữ liệu tích hợp:** Quét tự động từ `archive (4)` đến"
      " `archive (10)`"
  )
  st.write(
      f"📊 **Tổng dung lượng bản ghi sau ETL:** `{len(df):,} dòng dữ liệu cầu"
      " thủ`"
  )
  st.write(
      "💾 **Tầng lưu trữ (Storage):** Đang hoạt động trên bộ nhớ đệm `Streamlit"
      " Cache`"
  )
  st.write(
      "🤖 **Mô hình học máy:** Scikit-Learn Supervised Machine Learning"
      " (Linear Regression)"
  )

  st.divider()
  st.write("📋 **Xem trước 10 bản ghi Master Dataset:**")
  st.dataframe(df.head(10), width="stretch")