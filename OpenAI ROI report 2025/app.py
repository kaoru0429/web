import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import io

# ==========================================
# 1. 全局配置與 UI 設計 (UI/UX)
# ==========================================
st.set_page_config(
    page_title="OpenAI 財務戰情室",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 設定繪圖風格
plt.style.use('seaborn-v0_8-darkgrid')

# 中文字型處理 (支援 Windows, Mac, Linux/Cloud)
def get_chinese_font():
    system = platform.system()
    if system == 'Windows':
        return ['Microsoft JhengHei', 'SimHei']
    elif system == 'Darwin':  # Mac
        return ['Arial Unicode MS', 'PingFang TC']
    else:
        # Streamlit Cloud (Linux) 
        return ['Noto Sans CJK JP', 'WenQuanYi Zen Hei']

def configure_plotting():
    plt.rcParams['font.sans-serif'] = get_chinese_font()
    plt.rcParams['axes.unicode_minus'] = False

configure_plotting()

# 自定義 CSS 優化視覺體驗
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f9f9f9;
        border-radius: 5px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 側邊欄：控制面板 (Tools)
# ==========================================
with st.sidebar:
    st.title("🎛️ 參數控制台")
    st.markdown("---")
    
    st.header("1. 市場環境假設")
    years = st.slider("模擬年限 (Years)", 3, 10, 5, help="預測未來的時間跨度")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        initial_users = st.number_input("初始用戶 (百萬人)", value=50, step=1, help="Year 0 的付費用戶數") * 1_000_000
    with col_u2:
        user_growth_rate = st.number_input("年增長率 (%)", value=30, step=5) / 100
        
    revenue_per_user = st.slider("每用戶年營收 (ARPU, USD)", 20, 200, 60, help="訂閱費 + API 使用量的平均值")
    
    st.markdown("---")
    
    st.header("2. 成本結構設定")
    st.info("💡 預設值基於 GPT-5.2 研發成本推估")
    
    dense_capex = st.number_input("Dense 初始研發投入 (M USD)", value=2025.0)
    base_inference_cost = st.slider("Dense 基準推論成本 (USD/人/年)", 10, 100, 50, help="傳統架構下，服務一位用戶一年的算力成本")

    # 隱藏的進階參數 (使用 Expander 收納，保持介面整潔)
    with st.expander("⚙️ 進階架構參數 (工程師專用)"):
        st.write("設定不同架構的成本折扣係數 (相對於 Dense)")
        moe_cost_ratio = st.slider("MoE 訓練成本係數", 0.1, 1.0, 0.6)
        hybrid_cost_ratio = st.slider("Hybrid 訓練成本係數", 0.1, 1.0, 0.7)
        
        st.write("設定推論效率係數")
        moe_inf_ratio = st.slider("MoE 推論係數", 0.1, 1.0, 0.35)
        hybrid_inf_ratio = st.slider("Hybrid 推論係數", 0.1, 1.0, 0.45)

# ==========================================
# 3. 核心模擬邏輯 (Simulation Engine)
# ==========================================
def run_simulation():
    # 初始化
    initial_investment = {
        "Dense": dense_capex,
        "MoE": dense_capex * moe_cost_ratio,
        "Hybrid": dense_capex * hybrid_cost_ratio
    }
    
    inf_multipliers = {
        "Dense": 1.0,
        "MoE": moe_inf_ratio,
        "Hybrid": hybrid_inf_ratio
    }
    
    data = {"年分": np.arange(0, years + 1)}
    raw_profit_data = {} # 儲存每年的淨利數據供分析用
    
    for arch in initial_investment.keys():
        cash_flow = [-initial_investment[arch]]
        current_users = initial_users
        current_balance = -initial_investment[arch]
        yearly_profits = []
        
        for _ in range(1, years + 1):
            # 營收計算
            revenue = current_users * revenue_per_user / 1_000_000 # 轉百萬
            
            # 成本計算
            unit_cost = base_inference_cost * inf_multipliers[arch]
            inf_cost = current_users * unit_cost / 1_000_000 # 轉百萬
            
            # 毛利計算
            gross_profit = revenue - inf_cost
            current_balance += gross_profit
            
            cash_flow.append(current_balance)
            yearly_profits.append(gross_profit)
            
            # 成長
            current_users *= (1 + user_growth_rate)
        
        data[arch] = cash_flow
        raw_profit_data[arch] = yearly_profits
        
    return pd.DataFrame(data), initial_investment, inf_multipliers

df_result, init_inv, inf_mult = run_simulation()

# ==========================================
# 4. 主畫面內容 (Dashboard)
# ==========================================
st.title("🤖 OpenAI 營運策略與財務戰情室")
st.markdown(f"**分析主題**：GPT-5.2 世代架構決策 (Dense vs MoE vs Hybrid) | **模擬時長**：{years} 年")

# 頂部關鍵指標 (KPI Cards)
col1, col2, col3 = st.columns(3)
final_year_idx = -1

# 計算指標
dense_final = df_result['Dense'].iloc[final_year_idx]
hybrid_final = df_result['Hybrid'].iloc[final_year_idx]
roi_gap = hybrid_final - dense_final

with col1:
    st.metric(label="Dense 預估現金流 (Year End)", value=f"${dense_final:,.0f} M", delta="基準線", delta_color="off")
with col2:
    st.metric(label="Hybrid 預估現金流 (Year End)", value=f"${hybrid_final:,.0f} M", delta=f"比 Dense 多賺 ${roi_gap:,.0f} M")
with col3:
    # 計算回本年 (簡單估算)
    break_even_year = "未回本"
    for i, val in enumerate(df_result['Hybrid']):
        if val >= 0:
            break_even_year = f"第 {i} 年"
            break
    st.metric(label="Hybrid 預計回本時間", value=break_even_year, delta="推薦策略", delta_color="inverse")

# 分頁導航
tab1, tab2, tab3 = st.tabs(["📈 趨勢分析圖表", "🔥 風險熱力圖", "📥 報表匯出"])

# --- Tab 1: 核心圖表 ---
with tab1:
    st.subheader("不同架構下的累積現金流模擬")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 繪製線圖
    colors = {'Dense': '#FF6B6B', 'MoE': '#4ECDC4', 'Hybrid': '#FFD93D'}
    styles = {'Dense': '--', 'MoE': '-.', 'Hybrid': '-'}
    widths = {'Dense': 2, 'MoE': 2, 'Hybrid': 3.5}
    
    for col in ['Dense', 'MoE', 'Hybrid']:
        ax.plot(df_result['年分'], df_result[col], 
                label=col, color=colors[col], linestyle=styles[col], linewidth=widths[col])
        # 標註終點
        ax.text(years, df_result[col].iloc[-1], f" ${df_result[col].iloc[-1]:,.0f}M", 
                fontsize=10, verticalalignment='center', fontweight='bold')

    ax.axhline(0, color='black', linewidth=1.5, alpha=0.5) # 損益兩平線
    ax.text(0.1, 50, '損益兩平點 (Break-even)', color='black', fontsize=10)
    
    ax.set_title(f"累積現金流預測 ({years} 年)", fontsize=14, fontweight='bold')
    ax.set_xlabel("上線後年數")
    ax.set_ylabel("百萬美元 (M USD)")
    ax.legend(title="模型架構", loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    with st.expander("查看詳細年度數據"):
        st.dataframe(df_result.style.format("{:,.0f}"), use_container_width=True)

# --- Tab 2: 敏感度分析 ---
with tab2:
    st.subheader("敏感度分析：ARPU vs 推論成本 (Hybrid 架構)")
    st.markdown("此熱力圖顯示在不同 **每用戶營收 (X軸)** 與 **推論成本 (Y軸)** 組合下，Hybrid 架構的 **回本年限**。")
    st.markdown("🟢 **綠色** = 快速回本 (安全區) | 🔴 **紅色** = 難以回本 (危險區)")
    
    # 產生資料矩陣
    cost_range = np.linspace(base_inference_cost * 0.5, base_inference_cost * 1.5, 5)
    arpu_range = np.linspace(revenue_per_user * 0.5, revenue_per_user * 1.5, 5)
    
    heatmap_data = []
    hybrid_init = init_inv['Hybrid']
    hybrid_inf_factor = inf_mult['Hybrid']
    
    for c in cost_range:
        row = []
        for r in arpu_range:
            # 簡化計算：假設第一年的用戶規模來估算靜態回本期
            users = initial_users
            rev = users * r / 1_000_000
            cost = users * (c * hybrid_inf_factor) / 1_000_000
            net = rev - cost
            
            if net <= 0:
                years_req = 10 # 代表無限大/虧損
            else:
                years_req = hybrid_init / net
            row.append(years_req)
        heatmap_data.append(row)
        
    df_heat = pd.DataFrame(
        heatmap_data, 
        index=[f"${c:.0f}" for c in cost_range], 
        columns=[f"${r:.0f}" for r in arpu_range]
    )
    
    fig_h, ax_h = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_heat, annot=True, fmt=".1f", cmap="RdYlGn_r", cbar_kws={'label': '回本年限 (年)'}, ax=ax_h)
    ax_h.set_title("Hybrid 架構投資回收期矩陣")
    ax_h.set_xlabel("每用戶年營收 (ARPU)")
    ax_h.set_ylabel("基礎推論成本 (Base Cost)")
    
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.pyplot(fig_h)
    with col_h2:
        st.info("💡 **解讀建議**：\n若落入紅色區域，建議：\n1. 提高 API/訂閱定價\n2. 透過技術優化降低推論成本")

# --- Tab 3: 報表匯出 ---
with tab3:
    st.subheader("財務模型導出中心")
    st.write("生成包含所有參數設定與模擬結果的專業 Excel 報表，可直接用於財務會議。")
    
    def generate_excel():
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # 1. 模擬結果表
            df_result.to_excel(writer, sheet_name='現金流預測', index=False)
            
            # 2. 參數設定表
            params = {
                '模擬年限': years,
                '初始用戶數': initial_users,
                '年成長率': user_growth_rate,
                'ARPU': revenue_per_user,
                'Dense 初始投入': init_inv['Dense'],
                'Hybrid 初始投入': init_inv['Hybrid'],
                'Dense 推論成本': base_inference_cost,
                'Hybrid 推論係數': inf_mult['Hybrid']
            }
            pd.DataFrame(list(params.items()), columns=['參數項目', '設定值']).to_excel(writer, sheet_name='假設參數', index=False)
            
            # 3. 敏感度分析數據
            df_heat.to_excel(writer, sheet_name='敏感度分析矩陣')
            
        return output.getvalue()
        
    excel_file = generate_excel()
    
    st.download_button(
        label="📥 下載完整財務分析報告 (.xlsx)",
        data=excel_file,
        file_name='OpenAI_Financial_Strategy_Report_2025.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        type="primary" # 按鈕樣式
    )

# 頁尾
st.markdown("---")
st.caption("© 2025 AI Strategy Simulation | Created by Streamlit & Python | Data based on hypothetical scenarios.")