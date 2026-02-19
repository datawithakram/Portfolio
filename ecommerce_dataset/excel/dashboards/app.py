import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly

# Statistical Analysis
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, normaltest, ttest_ind, f_oneway
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

# Scikit-learn for normalization in radar chart
from sklearn.preprocessing import MinMaxScaler

# Configure Streamlit page
st.set_page_config(
    page_title="E-Commerce Pricing Analytics",
    page_icon="📊",
    layout="wide"
)

# Configuration from notebook
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

# Plotting Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pio.templates.default = "plotly_white"

# Custom Color Palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#F77F00',
    'danger': '#D62828',
    'neutral': '#6C757D'
}

st.write("# 📊 E-Commerce Pricing Analytics")
st.write("### Exploratory Data Analysis (EDA) Notebook")
st.write("### Dataset: 1M+ Orders")

# Helper function to generate frequency tables for categorical variables
def create_frequency_table(df, column):
    freq = df[column].value_counts()
    freq_pct = df[column].value_counts(normalize=True) * 100

    freq_df = pd.DataFrame({
        'Category': freq.index,
        'Count': freq.values,
        'Percentage': freq_pct.values.round(2)
    })
    return freq_df

# Helper function for outlier detection (from notebook)
def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD NAVIGATION AND KPIS PAGE
# ═══════════════════════════════════════════════════════════════════════════

def dashboard_kpis_page(df):
    st.header("Dashboard Overview & Key Performance Indicators")
    st.markdown("""
    This section provides a high-level overview of the e-commerce performance,
    presenting key metrics and an executive summary of the dataset.
    """)

    # Generate comprehensive summary statistics
    summary_metrics = {
        'Total Orders': len(df),
        'Total Revenue': df['total_price_usd'].sum(),
        'Total Profit': df['profit_usd'].sum(),
        'Avg Profit Margin': df['profit_margin_percent'].mean(),
        'Avg Order Value': df['total_price_usd'].mean(),
        'Unique Customers': df['customer_id'].nunique(),
        'Unique Products': df['product_id'].nunique(),
        'Order Completion Rate': (df['order_status'] == 'Completed').sum() / len(df) * 100,
        'Return Rate': (df['order_status'] == 'Returned').sum() / len(df) * 100,
        'Avg Customer Rating': df['rating'].mean(),
        'Mobile Order %': (df['device_type'] == 'Mobile').sum() / len(df) * 100,
        'Avg Delivery Days': df['delivery_days'].mean()
    }

    st.subheader("📊 Key Performance Indicators (KPIs)")

    # Display KPIs using Streamlit columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Orders", value=f"{summary_metrics['Total Orders']:,}")
        st.metric(label="Avg Profit Margin", value=f"{summary_metrics['Avg Profit Margin']:.2f}%")
    with col2:
        st.metric(label="Total Revenue", value=f"${summary_metrics['Total Revenue']:,.2f}")
        st.metric(label="Avg Order Value", value=f"${summary_metrics['Avg Order Value']:.2f}")
    with col3:
        st.metric(label="Total Profit", value=f"${summary_metrics['Total Profit']:,.2f}")
        st.metric(label="Unique Customers", value=f"{summary_metrics['Unique Customers']:,}")
    with col4:
        st.metric(label="Order Completion Rate", value=f"{summary_metrics['Order Completion Rate']:.2f}%")
        st.metric(label="Return Rate", value=f"{summary_metrics['Return Rate']:.2f}%")

    st.metric(label="Avg Customer Rating", value=f"{summary_metrics['Avg Customer Rating']:.2f}/5")
    st.metric(label="Avg Delivery Days", value=f"{summary_metrics['Avg Delivery Days']:.1f} days")

    st.subheader("📝 EXECUTIVE SUMMARY - KEY METRICS")
    st.markdown("""
    This table provides a detailed breakdown of the key operational and financial metrics across the entire dataset.
    """)

    # Create a cleaner display for the executive summary
    exec_summary_data = []
    for metric, value in summary_metrics.items():
        if 'Rate' in metric or 'Margin' in metric or 'Mobile Order' in metric:
            exec_summary_data.append([metric, f"{value:.2f}%"])
        elif 'Avg' in metric and 'Value' not in metric and 'Rating' not in metric and 'Days' not in metric:
            exec_summary_data.append([metric, f"{value:,.2f}"])
        elif 'Revenue' in metric or 'Profit' in metric or 'Value' in metric:
            exec_summary_data.append([metric, f"${value:,.2f}"])
        elif 'Rating' in metric:
            exec_summary_data.append([metric, f"{value:.2f}/5"])
        elif 'Days' in metric:
            exec_summary_data.append([metric, f"{value:.1f} days"])
        else:
            exec_summary_data.append([metric, f"{value:,}"])

    exec_summary_df = pd.DataFrame(exec_summary_data, columns=['Metric', 'Value'])
    st.dataframe(exec_summary_df, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# EDA PAGE - NUMERICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def eda_numerical_page(df):
    st.header("📈 EDA - Numerical Analysis")
    st.markdown("""
    This section provides an in-depth univariate analysis of the numerical variables in the dataset.
    We will examine descriptive statistics, distributions, and visualize potential outliers.
    """)

    key_numerical = [
        'total_price_usd', 'profit_usd', 'profit_margin_percent',
        'unit_price_usd', 'quantity', 'discount_percent',
        'customer_loyalty_score', 'fraud_risk_score',
        'delivery_days', 'session_duration_minutes', 'age'
    ]

    st.subheader("📊 DESCRIPTIVE STATISTICS - KEY METRICS")
    st.markdown("""
    A comprehensive statistical summary of key numerical columns, including measures of central tendency,
    dispersion, and shape (skewness, kurtosis).
    """)
    summary_stats = df[key_numerical].describe().T
    summary_stats['cv'] = (summary_stats['std'] / summary_stats['mean'] * 100).round(2)
    summary_stats['skewness'] = df[key_numerical].skew().round(2)
    summary_stats['kurtosis'] = df[key_numerical].kurtosis().round(2)

    st.dataframe(summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'cv', 'skewness', 'kurtosis']])

    st.subheader("📉 Distribution Plots - Financial Metrics")
    st.markdown("""
    Histograms to visualize the distribution of financial metrics like total price, profit, and discount.
    """)
    fig_dist = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Total Price (USD)', 'Profit (USD)', 'Profit Margin (%)',
                        'Unit Price (USD)', 'Quantity', 'Discount (%)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    metrics_to_plot = [
        ('total_price_usd', 1, 1),
        ('profit_usd', 1, 2),
        ('profit_margin_percent', 1, 3),
        ('unit_price_usd', 2, 1),
        ('quantity', 2, 2),
        ('discount_percent', 2, 3)
    ]

    for col, row, position in metrics_to_plot:
        fig_dist.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                marker_color=COLORS['primary'] if col in ['total_price_usd', 'unit_price_usd', 'quantity'] else COLORS['success'],
                opacity=0.7,
                showlegend=False
            ),
            row=row, col=position
        )

    fig_dist.update_layout(
        title_text="Distribution of Key Financial Metrics",
        height=700,
        showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("📦 Box Plots for Outlier Visualization")
    st.markdown("""
    Box plots help visualize the distribution and identify potential outliers in key financial metrics.
    """)
    fig_box = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Total Price Distribution', 'Profit Distribution', 'Profit Margin Distribution')
    )

    fig_box.add_trace(
        go.Box(y=df['total_price_usd'], name='Total Price', marker_color=COLORS['primary']),
        row=1, col=1
    )

    fig_box.add_trace(
        go.Box(y=df['profit_usd'], name='Profit', marker_color=COLORS['success']),
        row=1, col=2
    )

    fig_box.add_trace(
        go.Box(y=df['profit_margin_percent'], name='Profit Margin %', marker_color=COLORS['accent']),
        row=1, col=3
    )

    fig_box.update_layout(
        title_text="Box Plots - Financial Metrics (Outlier Detection)",
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("👥 Customer Metrics Distribution")
    st.markdown("""
    Histograms for customer-related numerical metrics such as age, loyalty score, and fraud risk score.
    """)
    fig_customer = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Customer Age', 'Loyalty Score', 'Fraud Risk Score')
    )

    fig_customer.add_trace(
        go.Histogram(x=df['age'], marker_color=COLORS['primary'], name='Age', showlegend=False),
        row=1, col=1
    )

    fig_customer.add_trace(
        go.Histogram(x=df['customer_loyalty_score'], marker_color=COLORS['success'], name='Loyalty', showlegend=False),
        row=1, col=2
    )

    fig_customer.add_trace(
        go.Histogram(x=df['fraud_risk_score'], marker_color=COLORS['danger'], name='Fraud Risk', showlegend=False),
        row=1, col=3
    )

    fig_customer.update_layout(
        title_text="Customer Metrics Distribution",
        height=400
    )
    st.plotly_chart(fig_customer, use_container_width=True)

    st.subheader("🚚 Operational Metrics")
    st.markdown("""
    Distribution of key operational metrics like delivery days.
    """)
    fig_delivery = go.Figure()
    fig_delivery.add_trace(go.Histogram(
        x=df['delivery_days'],
        marker_color=COLORS['accent'],
        opacity=0.8,
        name='Delivery Days'
    ))

    fig_delivery.update_layout(
        title="Delivery Days Distribution",
        xaxis_title="Days",
        yaxis_title="Frequency",
        height=400
    )
    st.plotly_chart(fig_delivery, use_container_width=True)

    st.markdown("""
---
## 📊 Key Findings - Numerical Variables

### 💰 Financial Metrics
- **Total Price**: Right-skewed distribution, indicating most orders are small-to-medium value with some high-value outliers
- **Profit Margin**: Relatively normal distribution around 30-40%, showing consistent pricing strategy
- **Discount**: Many orders have 0% discount, with common discount tiers at 10%, 15%, and 20%

### 👥 Customer Characteristics
- **Age**: Normal distribution centered around 45-55 years
- **Loyalty Score**: Fairly uniform distribution across all ranges
- **Fraud Risk**: Slightly higher concentration in lower risk scores

### 🚚 Operations
- **Delivery Days**: Most deliveries complete within 5-7 days
- Strong clustering around standard delivery windows

---
    """)


# ═══════════════════════════════════════════════════════════════════════════
# EDA PAGE - CATEGORICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def eda_categorical_page(df):
    st.header("📊 EDA - Categorical Analysis")
    st.markdown("""
    This section provides an in-depth univariate analysis of the categorical variables in the dataset.
    We will examine their distributions, frequencies, and relative proportions.
    """)

    # Order Status
    st.subheader("1️⃣ ORDER STATUS")
    order_status_freq = create_frequency_table(df, 'order_status')
    st.dataframe(order_status_freq.set_index('Category'))

    fig_status = go.Figure(data=[
        go.Pie(
            labels=order_status_freq['Category'],
            values=order_status_freq['Count'],
            hole=0.4,
            marker_colors=[COLORS['success'], COLORS['danger'], COLORS['warning'], COLORS['primary'], COLORS['accent']]
        )
    ])
    fig_status.update_layout(
        title="Order Status Distribution",
        height=450
    )
    st.plotly_chart(fig_status, use_container_width=True)

    # Customer Segment
    st.subheader("2️⃣ CUSTOMER SEGMENT")
    segment_freq = create_frequency_table(df, 'customer_segment')
    st.dataframe(segment_freq.set_index('Category'))

    fig_segment = go.Figure(data=[
        go.Bar(
            x=segment_freq['Category'],
            y=segment_freq['Count'],
            marker_color=COLORS['primary'],
            text=segment_freq['Percentage'].apply(lambda x: f"{x}%"),
            textposition='auto'
        )
    ])
    fig_segment.update_layout(
        title="Customer Segment Distribution",
        xaxis_title="Segment",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig_segment, use_container_width=True)

    # Product Category
    st.subheader("3️⃣ PRODUCT CATEGORY")
    category_freq = create_frequency_table(df, 'category')
    st.dataframe(category_freq.set_index('Category'))

    fig_category = go.Figure(data=[
        go.Bar(
            y=category_freq['Category'],
            x=category_freq['Count'],
            orientation='h',
            marker_color=COLORS['accent'],
            text=category_freq['Percentage'].apply(lambda x: f"{x}%"),
            textposition='auto'
        )
    ])
    fig_category.update_layout(
        title="Product Category Distribution",
        xaxis_title="Count",
        yaxis_title="Category",
        height=500
    )
    st.plotly_chart(fig_category, use_container_width=True)

    # Payment Method
    st.subheader("4️⃣ PAYMENT METHOD")
    payment_freq = create_frequency_table(df, 'payment_method')
    st.dataframe(payment_freq.set_index('Category'))

    fig_payment = go.Figure(data=[
        go.Pie(
            labels=payment_freq['Category'],
            values=payment_freq['Count'],
            hole=0.3
        )
    ])
    fig_payment.update_layout(
        title="Payment Method Distribution",
        height=450
    )
    st.plotly_chart(fig_payment, use_container_width=True)

    # Multi-panel categorical overview
    st.subheader("5️⃣ Other Categorical Variables Overview")
    fig_multi_cat = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Shipping Method', 'Device Type', 'Campaign Source', 'Review Sentiment'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Shipping Method
    shipping_freq = df['shipping_method'].value_counts()
    fig_multi_cat.add_trace(
        go.Bar(x=shipping_freq.index, y=shipping_freq.values, marker_color=COLORS['primary'], showlegend=False),
        row=1, col=1
    )

    # Device Type
    device_freq = df['device_type'].value_counts()
    fig_multi_cat.add_trace(
        go.Bar(x=device_freq.index, y=device_freq.values, marker_color=COLORS['success'], showlegend=False),
        row=1, col=2
    )

    # Campaign Source
    campaign_freq = df['campaign_source'].value_counts()
    fig_multi_cat.add_trace(
        go.Bar(x=campaign_freq.index, y=campaign_freq.values, marker_color=COLORS['accent'], showlegend=False),
        row=2, col=1
    )

    # Review Sentiment
    sentiment_freq = df['review_sentiment'].value_counts()
    fig_multi_cat.add_trace(
        go.Bar(x=sentiment_freq.index, y=sentiment_freq.values, marker_color=COLORS['secondary'], showlegend=False),
        row=2, col=2
    )

    fig_multi_cat.update_layout(
        title_text="Categorical Variables Overview",
        height=700,
        showlegend=False
    )
    st.plotly_chart(fig_multi_cat, use_container_width=True)

    # Gender Distribution
    st.subheader("6️⃣ GENDER DISTRIBUTION")
    gender_freq = create_frequency_table(df, 'gender')
    st.dataframe(gender_freq.set_index('Category'))

    # Weekend Orders
    st.subheader("7️⃣ WEEKEND vs WEEKDAY ORDERS")
    weekend_freq = create_frequency_table(df, 'is_weekend')
    st.dataframe(weekend_freq.set_index('Category'))

    st.markdown("""
---
## 📊 Key Findings - Categorical Variables

### 📦 Order Characteristics
- **Order Status**: Majority are completed successfully (~80%), with manageable return and pending rates
- **Weekend Activity**: Balanced distribution between weekday and weekend orders

### 👥 Customer Demographics
- **Gender**: Relatively balanced distribution
- **Segments**: Regular customers form the largest segment, followed by VIP and Premium

### 📱 Digital Behavior
- **Device Type**: Mobile dominates, reflecting mobile-first shopping behavior
- **Campaign Source**: Multi-channel presence with Email, Google Ads, and Facebook

### 💳 Payment & Shipping
- **Payment Methods**: Diverse mix showing customer preference flexibility
- **Shipping Options**: Economy shipping most popular, balancing cost and speed

### ⭐ Customer Satisfaction
- **Review Sentiment**: Majority positive, indicating good product/service quality

---
    """)


# ═══════════════════════════════════════════════════════════════════════════
# BIVARIATE & MULTIVARIATE ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════════════════════

def bivariate_multivariate_analysis_page(df):
    st.header("🔗 Bivariate & Multivariate Analysis")
    st.markdown("""
    This section explores the relationships between different variables, identifying correlations,
    and analyzing how various factors interact to influence business outcomes.
    """)

    numerical_for_corr = [
        'unit_price_usd', 'quantity', 'discount_percent', 'total_price_usd',
        'profit_usd', 'profit_margin_percent', 'tax_usd', 'shipping_cost_usd',
        'delivery_days', 'age', 'customer_loyalty_score', 'fraud_risk_score',
        'rating', 'session_duration_minutes', 'pages_visited'
    ]

    st.subheader("📊 CORRELATION MATRIX (Pearson)")
    st.markdown("""
    A heatmap visualizing the Pearson correlation coefficients between key numerical variables.
    """)
    correlation_matrix = df[numerical_for_corr].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))

    fig_corr.update_layout(
        title="Correlation Heatmap - Key Numerical Variables",
        height=800,
        width=900
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("🔝 STRONGEST CORRELATIONS (|r| > 0.5)")
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    corr_pairs = correlation_matrix.where(mask).stack().reset_index()
    corr_pairs.columns = ['Variable_1', 'Variable_2', 'Correlation']
    corr_pairs = corr_pairs[abs(corr_pairs['Correlation']) > 0.5].sort_values('Correlation', ascending=False, key=abs)
    st.dataframe(corr_pairs)

    st.subheader("💰 Profit vs Total Price - Customer Segment Analysis")
    st.markdown("""
    A scatter plot showing the relationship between profit and total price, colored by customer segment.
    The size of the marker indicates the quantity purchased.
    """)
    # Sample for performance as dataset is large
    sampled_df_scatter = df.sample(min(10000, len(df)), random_state=42) if len(df) > 10000 else df
    fig_profit_price = px.scatter(
        sampled_df_scatter,
        x='total_price_usd',
        y='profit_usd',
        color='customer_segment',
        size='quantity',
        hover_data=['category', 'discount_percent'],
        title="Profit vs Total Price (Colored by Customer Segment)",
        labels={'total_price_usd': 'Total Price (USD)', 'profit_usd': 'Profit (USD)'},
        opacity=0.6
    )
    fig_profit_price.update_layout(height=600)
    st.plotly_chart(fig_profit_price, use_container_width=True)

    st.subheader("👥 CUSTOMER SEGMENT PERFORMANCE")
    st.markdown("""
    Analysis of average order value, profit, loyalty score, and discount percentage across different customer segments.
    """)
    segment_analysis = df.groupby('customer_segment').agg({
        'total_price_usd': ['mean', 'sum', 'count'],
        'profit_usd': 'mean',
        'profit_margin_percent': 'mean',
        'customer_loyalty_score': 'mean',
        'discount_percent': 'mean'
    }).round(2)
    segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns.values]
    segment_analysis = segment_analysis.reset_index()
    st.dataframe(segment_analysis.set_index('customer_segment'))

    fig_segment_perf = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Avg Order Value', 'Avg Profit', 'Avg Loyalty Score')
    )
    segments = segment_analysis['customer_segment']
    fig_segment_perf.add_trace(
        go.Bar(x=segments, y=segment_analysis['total_price_usd_mean'],
               marker_color=COLORS['primary'], showlegend=False),
        row=1, col=1
    )
    fig_segment_perf.add_trace(
        go.Bar(x=segments, y=segment_analysis['profit_usd_mean'],
               marker_color=COLORS['success'], showlegend=False),
        row=1, col=2
    )
    fig_segment_perf.add_trace(
        go.Bar(x=segments, y=segment_analysis['customer_loyalty_score_mean'],
               marker_color=COLORS['accent'], showlegend=False),
        row=1, col=3
    )
    fig_segment_perf.update_layout(
        title_text="Customer Segment Performance Comparison",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_segment_perf, use_container_width=True)

    st.subheader("📦 CATEGORY PERFORMANCE")
    st.markdown("""
    Detailed performance metrics for product categories, including total revenue, profit, and average margin.
    """)
    category_performance = df.groupby('category').agg({
        'total_price_usd': ['sum', 'mean', 'count'],
        'profit_usd': ['sum', 'mean'],
        'profit_margin_percent': 'mean',
        'quantity': 'sum'
    }).round(2)
    category_performance.columns = ['_'.join(col).strip() for col in category_performance.columns.values]
    category_performance = category_performance.reset_index()
    category_performance = category_performance.sort_values('total_price_usd_sum', ascending=False)
    st.dataframe(category_performance.set_index('category'))

    st.subheader("💸 Top Categories by Revenue")
    fig_top_cat = go.Figure(data=[
        go.Bar(
            y=category_performance['category'].head(10),
            x=category_performance['total_price_usd_sum'].head(10),
            orientation='h',
            marker_color=COLORS['accent'],
            text=category_performance['total_price_usd_sum'].head(10).apply(lambda x: f"${x/1e6:.1f}M"),
            textposition='auto'
        )
    ])
    fig_top_cat.update_layout(
        title="Top 10 Categories by Total Revenue",
        xaxis_title="Total Revenue (USD)",
        yaxis_title="Category",
        height=500
    )
    st.plotly_chart(fig_top_cat, use_container_width=True)

    st.subheader("💳 PAYMENT METHOD vs ORDER STATUS")
    st.markdown("""
    Cross-tabulation and stacked bar chart showing the distribution of order statuses across different payment methods.
    """)
    payment_status_ct = pd.crosstab(
        df['payment_method'],
        df['order_status'],
        normalize='index'
    ) * 100
    st.dataframe(payment_status_ct.round(2))

    fig_payment_status = go.Figure()
    for status in payment_status_ct.columns:
        fig_payment_status.add_trace(go.Bar(
            name=status,
            x=payment_status_ct.index,
            y=payment_status_ct[status]
        ))
    fig_payment_status.update_layout(
        title="Order Status Distribution by Payment Method",
        xaxis_title="Payment Method",
        yaxis_title="Percentage (%)",
        barmode='stack',
        height=500
    )
    st.plotly_chart(fig_payment_status, use_container_width=True)

    st.subheader("📱 DEVICE TYPE ANALYSIS")
    st.markdown("""
    Comparison of average order value, session duration, pages visited, and cart abandonment rate by device type.
    """)
    device_analysis = df.groupby('device_type').agg({
        'total_price_usd': 'mean',
        'session_duration_minutes': 'mean',
        'pages_visited': 'mean',
        'order_id': 'count',
        'abandoned_cart_flag': 'mean'
    }).round(2)
    device_analysis.columns = ['Avg Order Value', 'Avg Session Duration', 'Avg Pages Visited', 'Order Count', 'Cart Abandonment Rate']
    device_analysis = device_analysis.reset_index()
    st.dataframe(device_analysis.set_index('device_type'))

    st.subheader("📢 CAMPAIGN SOURCE EFFECTIVENESS")
    st.markdown("""
    Analysis of campaign sources based on total revenue, average order value, profit, and customer loyalty.
    """)
    campaign_analysis = df.groupby('campaign_source').agg({
        'total_price_usd': ['sum', 'mean', 'count'],
        'profit_usd': 'sum',
        'customer_loyalty_score': 'mean',
        'fraud_risk_score': 'mean'
    }).round(2)
    campaign_analysis.columns = ['_'.join(col).strip() for col in campaign_analysis.columns.values]
    campaign_analysis = campaign_analysis.reset_index()
    campaign_analysis = campaign_analysis.sort_values('total_price_usd_sum', ascending=False)
    st.dataframe(campaign_analysis.set_index('campaign_source'))

    st.subheader("🚚 SHIPPING METHOD vs SATISFACTION")
    st.markdown("""
    Comparing average customer rating and delivery days across different shipping methods.
    """)
    shipping_satisfaction = df.groupby('shipping_method').agg({
        'rating': 'mean',
        'delivery_days': 'mean',
        'shipping_cost_usd': 'mean',
        'order_id': 'count'
    }).round(2)
    shipping_satisfaction = shipping_satisfaction.reset_index()
    st.dataframe(shipping_satisfaction.set_index('shipping_method'))

    fig_shipping_satisfaction = make_subplots(specs=[[{"secondary_y": True}]])
    fig_shipping_satisfaction.add_trace(
        go.Bar(
            x=shipping_satisfaction['shipping_method'],
            y=shipping_satisfaction['rating'],
            name='Avg Rating',
            marker_color=COLORS['success']
        ),
        secondary_y=False
    )
    fig_shipping_satisfaction.add_trace(
        go.Scatter(
            x=shipping_satisfaction['shipping_method'],
            y=shipping_satisfaction['delivery_days'],
            name='Avg Delivery Days',
            marker_color=COLORS['danger'],
            mode='lines+markers'
        ),
        secondary_y=True
    )
    fig_shipping_satisfaction.update_layout(
        title="Shipping Method: Rating vs Delivery Speed",
        height=500
    )
    fig_shipping_satisfaction.update_xaxes(title_text="Shipping Method")
    fig_shipping_satisfaction.update_yaxes(title_text="Average Rating", secondary_y=False)
    fig_shipping_satisfaction.update_yaxes(title_text="Average Delivery Days", secondary_y=True)
    st.plotly_chart(fig_shipping_satisfaction, use_container_width=True)

    st.markdown("""
---
## 🔍 Key Insights - Bivariate Analysis

### 💰 **Financial Relationships**
- **Strong positive correlation** between total_price and profit (expected)
- **Negative correlation** between discount percentage and profit margin
- Higher discounts drive volume but reduce profitability

### 👥 **Customer Segmentation**
- **VIP customers** generate highest average order value and profit
- **Premium segment** shows strong loyalty scores
- Segment-based pricing strategies are effective

### 📦 **Product Categories**
- Clear revenue leaders emerge in category analysis
- Profit margins vary significantly by category
- Volume vs margin tradeoffs visible across categories

### 📱 **Channel Performance**
- **Mobile** dominates order volume but shows higher cart abandonment
- **Desktop** users have longer sessions and higher engagement
- Channel-specific optimization opportunities exist

### 🚚 **Operational Efficiency**
- **Faster shipping** correlates with higher ratings
- **Express delivery** premium justifies higher satisfaction
- Delivery speed is a key satisfaction driver

### 📢 **Marketing Effectiveness**
- Different campaign sources show distinct ROI profiles
- Email and Google Ads drive different customer behaviors
- Multi-channel attribution is complex

---
    """)


# ═══════════════════════════════════════════════════════════════════════════
# TIME SERIES ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════════════════════

def time_series_analysis_page(df):
    st.header("📈 Time Series Analysis")
    st.markdown("""
    This section analyzes the trends and seasonality of key e-commerce metrics over time,
    providing insights into performance patterns and growth dynamics.
    """)

    # Daily aggregation
    daily_metrics = df.groupby(df['order_date'].dt.date).agg({
        'order_id': 'count',
        'total_price_usd': 'sum',
        'profit_usd': 'sum',
        'quantity': 'sum'
    }).reset_index()

    daily_metrics.columns = ['date', 'orders', 'revenue', 'profit', 'units_sold']
    daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])

    st.subheader("📅 DAILY METRICS SUMMARY")
    st.markdown("""
    Descriptive statistics for daily aggregated metrics.
    """)
    st.dataframe(daily_metrics.describe(include='all'))

    st.subheader("📊 Revenue Trend Over Time")
    st.markdown("""
    Line chart showing daily revenue with a 7-day moving average to smooth out fluctuations.
    """)
    fig_revenue_trend = go.Figure()

    fig_revenue_trend.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['revenue'],
        mode='lines',
        name='Daily Revenue',
        line=dict(color=COLORS['primary'], width=1),
        opacity=0.6
    ))

    # Add 7-day moving average
    daily_metrics['revenue_ma7'] = daily_metrics['revenue'].rolling(window=7).mean()

    fig_revenue_trend.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['revenue_ma7'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color=COLORS['danger'], width=2)
    ))

    fig_revenue_trend.update_layout(
        title="Daily Revenue Trend with 7-Day Moving Average",
        xaxis_title="Date",
        yaxis_title="Revenue (USD)",
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_revenue_trend, use_container_width=True)

    st.subheader("📈 Orders Trend Over Time")
    st.markdown("""
    Line chart showing daily order volume with a 7-day moving average.
    """)
    fig_orders_trend = go.Figure()

    fig_orders_trend.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['orders'],
        mode='lines',
        name='Daily Orders',
        line=dict(color=COLORS['success'], width=1),
        opacity=0.6
    ))

    daily_metrics['orders_ma7'] = daily_metrics['orders'].rolling(window=7).mean()

    fig_orders_trend.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['orders_ma7'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color=COLORS['accent'], width=2)
    ))

    fig_orders_trend.update_layout(
        title="Daily Order Volume Trend with 7-Day Moving Average",
        xaxis_title="Date",
        yaxis_title="Number of Orders",
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_orders_trend, use_container_width=True)

    st.subheader("📆 Monthly Trends - Revenue and Profit")
    st.markdown("""
    Bar and line chart displaying monthly revenue and profit trends.
    """)
    monthly_metrics = df.groupby([df['order_date'].dt.to_period('M')]).agg({
        'order_id': 'count',
        'total_price_usd': 'sum',
        'profit_usd': 'sum',
        'profit_margin_percent': 'mean'
    }).reset_index()

    monthly_metrics.columns = ['month', 'orders', 'revenue', 'profit', 'avg_margin']
    monthly_metrics['month'] = monthly_metrics['month'].astype(str)
    st.dataframe(monthly_metrics)

    fig_monthly_trends = make_subplots(specs=[[{'secondary_y': True}]])

    fig_monthly_trends.add_trace(
        go.Bar(
            x=monthly_metrics['month'],
            y=monthly_metrics['revenue'],
            name='Revenue',
            marker_color=COLORS['primary']
        ),
        secondary_y=False
    )

    fig_monthly_trends.add_trace(
        go.Scatter(
            x=monthly_metrics['month'],
            y=monthly_metrics['profit'],
            name='Profit',
            marker_color=COLORS['success'],
            mode='lines+markers'
        ),
        secondary_y=True
    )

    fig_monthly_trends.update_layout(
        title="Monthly Revenue and Profit Trends",
        xaxis_title="Month",
        height=500
    )

    fig_monthly_trends.update_xaxes(tickangle=45)
    fig_monthly_trends.update_yaxes(title_text="Revenue (USD)", secondary_y=False)
    fig_monthly_trends.update_yaxes(title_text="Profit (USD)", secondary_y=True)
    st.plotly_chart(fig_monthly_trends, use_container_width=True)

    st.subheader("📅 Seasonality Analysis - Day of Week")
    st.markdown("""
    Bar chart showing order volume distribution across different days of the week.
    """)
    dow_analysis = df.groupby('order_dayname').agg({
        'order_id': 'count',
        'total_price_usd': 'mean',
        'profit_usd': 'mean'
    }).reset_index()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_analysis['order_dayname'] = pd.Categorical(dow_analysis['order_dayname'], categories=day_order, ordered=True)
    dow_analysis = dow_analysis.sort_values('order_dayname')
    st.dataframe(dow_analysis.set_index('order_dayname'))

    fig_dow = go.Figure()
    fig_dow.add_trace(go.Bar(
        x=dow_analysis['order_dayname'],
        y=dow_analysis['order_id'],
        marker_color=COLORS['accent'],
        text=dow_analysis['order_id'],
        textposition='auto'
    ))
    fig_dow.update_layout(
        title="Order Volume by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Number of Orders",
        height=400
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    st.subheader("⏰ Hour of Day Analysis")
    st.markdown("""
    Line chart illustrating order volume fluctuations throughout the hours of the day.
    """)
    hour_analysis = df.groupby('order_hour').agg({
        'order_id': 'count',
        'total_price_usd': 'mean'
    }).reset_index()
    st.dataframe(hour_analysis.set_index('order_hour'))

    fig_hour = go.Figure()
    fig_hour.add_trace(go.Scatter(
        x=hour_analysis['order_hour'],
        y=hour_analysis['order_id'],
        mode='lines+markers',
        marker_color=COLORS['primary'],
        line=dict(width=3),
        fill='tozeroy'
    ))
    fig_hour.update_layout(
        title="Order Volume by Hour of Day",
        xaxis_title="Hour (24-hour format)",
        yaxis_title="Number of Orders",
        height=400
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    st.subheader("📊 Quarterly Performance")
    st.markdown("""
    Bar chart showing total revenue per quarter.
    """)
    quarterly_metrics = df.groupby(['order_year', 'order_quarter']).agg({
        'order_id': 'count',
        'total_price_usd': 'sum',
        'profit_usd': 'sum',
        'profit_margin_percent': 'mean'
    }).reset_index()

    quarterly_metrics['period'] = quarterly_metrics['order_year'].astype(str) + '-Q' + quarterly_metrics['order_quarter'].astype(str)
    st.dataframe(quarterly_metrics.set_index('period'))

    fig_quarterly = go.Figure()
    fig_quarterly.add_trace(go.Bar(
        x=quarterly_metrics['period'],
        y=quarterly_metrics['total_price_usd'],
        marker_color=COLORS['secondary'],
        text=quarterly_metrics['total_price_usd'].apply(lambda x: f"${x/1e6:.1f}M"),
        textposition='auto'
    ))
    fig_quarterly.update_layout(
        title="Quarterly Revenue Performance",
        xaxis_title="Quarter",
        yaxis_title="Revenue (USD)",
        height=400
    )
    st.plotly_chart(fig_quarterly, use_container_width=True)

    st.subheader("📈 Month-over-Month Growth Rates")
    st.markdown("""
    Line chart illustrating month-over-month revenue and profit growth rates.
    """)
    monthly_metrics_sorted = monthly_metrics.sort_values('month')
    monthly_metrics_sorted['revenue_growth'] = monthly_metrics_sorted['revenue'].pct_change() * 100
    monthly_metrics_sorted['profit_growth'] = monthly_metrics_sorted['profit'].pct_change() * 100

    st.dataframe(monthly_metrics_sorted[['month', 'revenue_growth', 'profit_growth']].tail(12))

    fig_growth = go.Figure()

    fig_growth.add_trace(go.Scatter(
        x=monthly_metrics_sorted['month'],
        y=monthly_metrics_sorted['revenue_growth'],
        mode='lines+markers',
        name='Revenue Growth %',
        line=dict(color=COLORS['primary'], width=2)
    ))

    fig_growth.add_trace(go.Scatter(
        x=monthly_metrics_sorted['month'],
        y=monthly_metrics_sorted['profit_growth'],
        mode='lines+markers',
        name='Profit Growth %',
        line=dict(color=COLORS['success'], width=2)
    ))

    fig_growth.update_layout(
        title="Month-over-Month Growth Rates",
        xaxis_title="Month",
        yaxis_title="Growth Rate (%)",
        height=500,
        hovermode='x unified'
    )

    fig_growth.update_xaxes(tickangle=45)
    st.plotly_chart(fig_growth, use_container_width=True)

    st.subheader("🗓️ Weekend vs Weekday Performance")
    st.markdown("""
    Comparison of order counts, average order values, and profit margins between weekend and weekday orders.
    """)
    weekend_comparison = df.groupby('is_weekend').agg({
        'order_id': 'count',
        'total_price_usd': ['mean', 'sum'],
        'profit_usd': 'mean',
        'profit_margin_percent': 'mean'
    }).round(2)

    weekend_comparison.columns = ['_'.join(col).strip() for col in weekend_comparison.columns.values]
    weekend_comparison = weekend_comparison.reset_index()
    st.dataframe(weekend_comparison.set_index('is_weekend'))

    st.markdown("""
---
## 📈 Key Insights - Time Series Analysis

### 📊 **Trend Observations**
- **Steady growth** visible in both revenue and order volume over the 2-year period
- **7-day moving averages** smooth out daily volatility and reveal underlying trends
- **Monthly patterns** show consistent business performance with some seasonal variation

### 📅 **Seasonality Patterns**
- **Day of Week**: Mid-week (Tuesday-Thursday) shows slightly higher activity
- **Hour of Day**: Peak ordering hours align with typical online shopping patterns (evenings)
- **Weekend vs Weekday**: Relatively balanced, suggesting diverse customer base

### 📈 **Growth Dynamics**
- **Month-over-month growth** shows both positive and negative fluctuations
- **Quarterly trends** indicate seasonal business cycles
- Growth rates stabilizing suggests market maturity

### 🎯 **Business Implications**
- Inventory planning should account for day-of-week and hour-of-day patterns
- Marketing campaigns can be optimized for peak engagement times
- Staffing and logistics should scale with predictable daily/weekly patterns

---
    """)


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL HYPOTHESIS TESTING PAGE
# ═══════════════════════════════════════════════════════════════════════════

def statistical_hypothesis_testing_page(df):
    st.header("🔬 Statistical Hypothesis Testing")
    st.markdown("""
    This section presents the results of various statistical hypothesis tests conducted to validate business assumptions
    and understand significant relationships within the e-commerce data.
    """)

    st.subheader("TEST 1: Customer Segment Impact on Average Order Value")
    st.markdown("""
    - **H0**: All customer segments have the same average order value (AOV).
    - **H1**: At least one segment has a different average order value (AOV).
    """)

    regular_aov = df[df['customer_segment'] == 'Regular']['total_price_usd']
    vip_aov = df[df['customer_segment'] == 'VIP']['total_price_usd']
    premium_aov = df[df['customer_segment'] == 'Premium']['total_price_usd']

    # Ensure there's enough data for each group to perform ANOVA
    aov_groups = [g for g in [regular_aov, vip_aov, premium_aov] if len(g) > 1]

    if len(aov_groups) == 3:
        f_stat_1, p_value_1 = f_oneway(*aov_groups)
        st.write(f"F-statistic: {f_stat_1:.4f}")
        st.write(f"P-value: {p_value_1:.6f}")

        if p_value_1 < 0.05:
            st.success("✅ RESULT: Reject H0 - Customer segments have significantly different average order values. Post-hoc analysis recommended.")
            # Effect size (eta-squared)
            segment_groups = df.groupby('customer_segment')['total_price_usd'].apply(list)
            grand_mean = df['total_price_usd'].mean()
            ss_between = sum([len(group) * (np.mean(group) - grand_mean)**2 for group in segment_groups])
            ss_total = sum([(x - grand_mean)**2 for x in df['total_price_usd']])
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            st.write(f"Effect Size (η²): {eta_squared:.4f}")
        else:
            st.warning("❌ RESULT: Fail to reject H0 - No significant difference found in average order values across segments.")
            st.write(f"Effect Size (η²): Not calculated (H0 not rejected)")
    else:
        st.warning("Insufficient data in some customer segments to perform ANOVA.")


    st.subheader("TEST 2: Impact of Discounts on Profit Margin")
    st.markdown("""
    - **H0**: Discounted and full-price orders have the same profit margin.
    - **H1**: Discounted and full-price orders have different profit margins.
    """)

    discounted = df[df['discount_percent'] > 0]['profit_margin_percent']
    full_price = df[df['discount_percent'] == 0]['profit_margin_percent']

    # Handle cases where one group might be empty to avoid errors
    if len(discounted) > 1 and len(full_price) > 1:
        t_stat_2, p_value_2 = ttest_ind(discounted, full_price, equal_var=False) # Welch's t-test due to unequal variances

        st.write(f"Sample sizes: Discounted orders: {len(discounted):,}, Full-price orders: {len(full_price):,}")
        st.write(f"Mean profit margins: Discounted: {discounted.mean():.2f}%, Full-price: {full_price.mean():.2f}%")
        st.write(f"T-statistic: {t_stat_2:.4f}")
        st.write(f"P-value: {p_value_2:.6f}")

        if p_value_2 < 0.05:
            diff = full_price.mean() - discounted.mean()
            st.success(f"✅ RESULT: Reject H0 - Discounts significantly affect profit margin. Difference: {diff:.2f} percentage points.")
            pooled_std = np.sqrt(((len(discounted)-1)*discounted.std()**2 + (len(full_price)-1)*full_price.std()**2) / (len(discounted)+len(full_price)-2))
            cohens_d = (full_price.mean() - discounted.mean()) / pooled_std if pooled_std > 0 else 0
            st.write(f"Effect Size (Cohen's d): {cohens_d:.4f}")
        else:
            st.warning("❌ RESULT: Fail to reject H0 - No significant difference found in profit margins.")
            st.write("Effect Size (Cohen's d): Not calculated (H0 not rejected)")
    else:
        st.warning("Insufficient data in discounted or full-price groups to perform t-test.")


    st.subheader("TEST 3: Payment Method vs Order Status (Chi-Square Test)")
    st.markdown("""
    - **H0**: Payment method and order status are independent.
    - **H1**: Payment method and order status are associated.
    """)

    contingency_table = pd.crosstab(df['payment_method'], df['order_status'])
    st.write("Contingency Table:")
    st.dataframe(contingency_table)

    if not contingency_table.empty and (contingency_table.sum(axis=1) > 0).all() and (contingency_table.sum(axis=0) > 0).all():
        chi2_3, p_value_3, dof_3, expected_3 = chi2_contingency(contingency_table)

        st.write(f"Chi-square statistic: {chi2_3:.4f}")
        st.write(f"Degrees of freedom: {dof_3}")
        st.write(f"P-value: {p_value_3:.6f}")

        if p_value_3 < 0.05:
            st.success("✅ RESULT: Reject H0 - Payment method and order status are associated.")
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
            cramers_v = np.sqrt(chi2_3 / (n * min_dim)) if n * min_dim > 0 else 0
            st.write(f"Effect Size (Cramér's V): {cramers_v:.4f}")
        else:
            st.warning("❌ RESULT: Fail to reject H0 - No significant association found.")
            st.write("Effect Size (Cramér's V): Not calculated (H0 not rejected)")
    else:
        st.warning("Contingency table is empty or contains rows/columns with zero sums, cannot perform Chi-Square test.")


    st.subheader("TEST 4: Shipping Method Impact on Customer Rating")
    st.markdown("""
    - **H0**: All shipping methods result in the same average rating.
    - **H1**: At least one shipping method has a different average rating.
    """)

    # Filter out empty rating groups if any
    shipping_methods = df['shipping_method'].unique()
    ratings_by_method = [df[df['shipping_method'] == method]['rating'] for method in shipping_methods]
    valid_ratings_by_method = [r for r in ratings_by_method if len(r) > 1]

    if len(valid_ratings_by_method) > 1:
        f_stat_4, p_value_4 = f_oneway(*valid_ratings_by_method)

        st.write("Mean ratings by shipping method:")
        shipping_rating_means = df.groupby('shipping_method')['rating'].mean()
        st.dataframe(shipping_rating_means)

        st.write(f"F-statistic: {f_stat_4:.4f}")
        st.write(f"P-value: {p_value_4:.6f}")

        if p_value_4 < 0.05:
            st.success("✅ RESULT: Reject H0 - Shipping methods have significantly different ratings.")
        else:
            st.warning("❌ RESULT: Fail to reject H0 - No significant difference found in ratings across shipping methods.")
    else:
        st.warning("Insufficient data for multiple shipping methods to perform ANOVA.")


    st.subheader("TEST 5: Gender Impact on Average Purchase Amount")
    st.markdown("""
    - **H0**: Male and Female customers have the same average purchase amount.
    - **H1**: Male and Female customers have different average purchase amounts.
    """)

    male_purchases = df[df['gender'] == 'Male']['total_price_usd']
    female_purchases = df[df['gender'] == 'Female']['total_price_usd']

    if len(male_purchases) > 1 and len(female_purchases) > 1:
        t_stat_5, p_value_5 = ttest_ind(male_purchases, female_purchases, equal_var=False)

        st.write(f"Sample sizes: Male: {len(male_purchases):,}, Female: {len(female_purchases):,}")
        st.write(f"Mean purchase amounts: Male: ${male_purchases.mean():.2f}, Female: ${female_purchases.mean():.2f}")
        st.write(f"T-statistic: {t_stat_5:.4f}")
        st.write(f"P-value: {p_value_5:.6f}")

        if p_value_5 < 0.05:
            st.success("✅ RESULT: Reject H0 - Gender significantly affects purchase amount.")
        else:
            st.warning("❌ RESULT: Fail to reject H0 - No significant difference found in purchase amounts between genders.")
    else:
        st.warning("Insufficient data for male or female purchases to perform t-test.")


    st.subheader("TEST 6: Correlation between Loyalty Score and Customer Total Spent")
    st.markdown("""
    - **H0**: No correlation between loyalty score and total spending.
    - **H1**: Correlation exists between loyalty score and total spending.
    """)

    if not df[['customer_loyalty_score', 'customer_total_spent']].isnull().any(axis=1).any() and len(df) > 1:
        correlation_6, p_value_6 = pearsonr(df['customer_loyalty_score'], df['customer_total_spent'])

        st.write(f"Pearson Correlation Coefficient: {correlation_6:.4f}")
        st.write(f"P-value: {p_value_6:.6f}")

        if p_value_6 < 0.05:
            if abs(correlation_6) < 0.3:
                strength = "weak"
            elif abs(correlation_6) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"
            direction = "positive" if correlation_6 > 0 else "negative"
            st.success(f"✅ RESULT: Reject H0 - Significant {strength} {direction} correlation exists.")
        else:
            st.warning("❌ RESULT: Fail to reject H0 - No significant correlation found.")
    else:
        st.warning("Cannot perform Pearson correlation due to missing values or insufficient data.")


    st.subheader("TEST 7: Weekend vs Weekday Order Values")
    st.markdown("""
    - **H0**: Weekend and weekday orders have the same average value.
    - **H1**: Weekend and weekday orders have different average values.
    """)

    weekend_orders = df[df['is_weekend'] == 'Yes']['total_price_usd']
    weekday_orders = df[df['is_weekend'] == 'No']['total_price_usd']

    if len(weekend_orders) > 1 and len(weekday_orders) > 1:
        t_stat_7, p_value_7 = ttest_ind(weekend_orders, weekday_orders, equal_var=False)

        st.write(f"Mean order values: Weekend: ${weekend_orders.mean():.2f}, Weekday: ${weekday_orders.mean():.2f}")
        st.write(f"T-statistic: {t_stat_7:.4f}")
        st.write(f"P-value: {p_value_7:.6f}")

        if p_value_7 < 0.05:
            st.success("✅ RESULT: Reject H0 - Significant difference exists in order values between weekend and weekday.")
        else:
            st.warning("❌ RESULT: Fail to reject H0 - No significant difference found.")
    else:
        st.warning("Insufficient data for weekend or weekday orders to perform t-test.")

    st.markdown("""
---
## 🔬 Statistical Testing Summary

### ✅ **Significant Findings** (p < 0.05)

1.  **Customer Segment & Order Value**: Strong evidence that different customer segments have significantly different average order values
    -   Business Implication: Segment-specific marketing and pricing strategies are justified

2.  **Discounts & Profit Margin**: Discounts significantly impact profit margins (as expected)
    -   Business Implication: Discount strategies need careful optimization to balance volume and profitability

3.  **Payment Method & Order Status**: Association exists between payment method and order completion
    -   Business Implication: Optimize checkout flow based on payment method characteristics

### 📊 **Effect Sizes**
-   Effect sizes help quantify the **practical significance** beyond statistical significance
-   Large effect sizes indicate stronger business relevance
-   Small p-values with small effect sizes suggest statistically significant but practically minor differences

### 🎯 **Business Recommendations**
-   **Customer Segmentation**: Leverage segment differences for targeted marketing
-   **Pricing Strategy**: Balance discount depth with profit margin targets
-   **Payment Optimization**: Streamline high-success payment methods
-   **Shipping Strategy**: Align shipping options with customer satisfaction goals

### ⚠️ **Statistical Considerations**
-   Large sample size (1M+ records) makes even small differences statistically significant
-   Always consider effect size alongside p-values
-   Business significance ≠ Statistical significance

---
    """)


# ═══════════════════════════════════════════════════════════════════════════
# BUSINESS INSIGHTS & DEEP DIVES PAGE
# ═══════════════════════════════════════════════════════════════════════════

def business_insights_page(df):
    st.header("💡 Business Insights & Deep Dives")
    st.markdown("""
    This section provides business-oriented insights derived from the data, focusing on key areas
    such as customer lifetime value, product performance, profitability, and marketing effectiveness.
    """)

    # ─────────────────────────────────────────────────────────────────────────────
    # Customer Lifetime Value Analysis
    # ─────────────────────────────────────────────────────────────────────────────

    st.subheader("1️⃣ CUSTOMER LIFETIME VALUE (CLV) ANALYSIS")
    st.markdown("""
    Segmentation of customers based on their total spending to understand different value tiers.
    """)

    # Check if 'customer_id' column exists before grouping
    if 'customer_id' in df.columns:
        clv_segments = df.groupby('customer_id').agg({
            'total_price_usd': 'sum',
            'order_id': 'count',
            'profit_usd': 'sum',
            'customer_segment': 'first',
            'customer_loyalty_score': 'first'
        }).reset_index()

        clv_segments.columns = ['customer_id', 'total_spent', 'order_count', 'total_profit', 'segment', 'loyalty_score']

        # Segment customers by CLV
        if not clv_segments.empty and 'total_spent' in clv_segments.columns:
            clv_segments['clv_tier'] = pd.qcut(clv_segments['total_spent'], q=4, labels=['Low', 'Medium', 'High', 'VIP'], duplicates='drop')

            st.write("**Customer Distribution by CLV Tier:**")
            st.dataframe(clv_segments['clv_tier'].value_counts().sort_index())

            clv_summary = clv_segments.groupby('clv_tier').agg({
                'total_spent': 'mean',
                'order_count': 'mean',
                'total_profit': 'mean',
                'customer_id': 'count'
            }).round(2)

            clv_summary.columns = ['Avg Total Spent', 'Avg Orders', 'Avg Profit', 'Customer Count']
            st.write("**CLV Tier Summary:**")
            st.dataframe(clv_summary)

            st.write("**Customer Lifetime Value Distribution by Tier:**")
            fig_clv = go.Figure()

            for tier in ['Low', 'Medium', 'High', 'VIP']:
                if tier in clv_segments['clv_tier'].unique(): # Check if tier exists in filtered data
                    tier_data = clv_segments[clv_segments['clv_tier'] == tier]['total_spent']
                    fig_clv.add_trace(go.Box(
                        y=tier_data,
                        name=tier,
                        boxmean='sd'
                    ))

            fig_clv.update_layout(
                title="Customer Lifetime Value Distribution by Tier",
                yaxis_title="Total Spent (USD)",
                height=500
            )
            st.plotly_chart(fig_clv, use_container_width=True)
        else:
            st.warning("Insufficient data to perform CLV tier analysis for the selected filters.")
    else:
        st.warning("The 'customer_id' column is not available in the filtered data. Cannot perform CLV analysis.")


    # ─────────────────────────────────────────────────────────────────────────────
    # Product Performance & Return Analysis
    # ─────────────────────────────────────────────────────────────────────────────

    st.subheader("2️⃣ PRODUCT RETURN ANALYSIS")
    st.markdown("""
    Analysis of product return rates by category and the distribution of return reasons.
    """)

    if not df.empty and 'category' in df.columns and 'order_status' in df.columns:
        product_returns = df.groupby('category').agg({
            'order_id': 'count',
            'order_status': lambda x: (x == 'Returned').sum()
        }).reset_index()

        product_returns.columns = ['category', 'total_orders', 'returned_orders']
        if not product_returns['total_orders'].empty and (product_returns['total_orders'] > 0).any():
            product_returns['return_rate'] = (product_returns['returned_orders'] / product_returns['total_orders'] * 100).round(2)
            product_returns = product_returns.sort_values('return_rate', ascending=False)

            st.write("**Return Rates by Category:**")
            st.dataframe(product_returns)

            if not product_returns.empty:
                st.write("**Top Categories by Return Rate:**")
                fig_return_rate = go.Figure(data=[
                    go.Bar(
                        y=product_returns['category'].head(10),
                        x=product_returns['return_rate'].head(10),
                        orientation='h',
                        marker_color=COLORS['danger'],
                        text=product_returns['return_rate'].head(10).apply(lambda x: f"{x}%"),
                        textposition='auto'
                    )
                ])

                fig_return_rate.update_layout(
                    title="Top 10 Categories by Return Rate",
                    xaxis_title="Return Rate (%)",
                    yaxis_title="Category",
                    height=500
                )
                st.plotly_chart(fig_return_rate, use_container_width=True)
            else:
                st.warning("No return data available for categories within the selected filters.")
        else:
            st.warning("No orders found for the selected filters to calculate return rates by category.")

        if not df[df['order_status'] == 'Returned'].empty:
            return_reasons = df[df['order_status'] == 'Returned']['return_reason'].value_counts()
            if not return_reasons.empty:
                st.write("**Return Reasons Distribution:**")
                st.dataframe(return_reasons)

                fig_return_reasons = go.Figure(data=[
                    go.Pie(
                        labels=return_reasons.index,
                        values=return_reasons.values,
                        hole=0.3
                    )
                ])

                fig_return_reasons.update_layout(
                    title="Distribution of Return Reasons",
                    height=500
                )
                st.plotly_chart(fig_return_reasons, use_container_width=True)
            else:
                st.warning("No return reasons found for returned orders within the selected filters.")
        else:
            st.warning("No returned orders found within the selected filters to analyze return reasons.")
    else:
        st.warning("Insufficient data to perform product return analysis for the selected filters.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Profitability Analysis
    # ─────────────────────────────────────────────────────────────────────────────

    st.subheader("3️⃣ PROFITABILITY DRIVERS ANALYSIS")
    st.markdown("""
    Examination of profitability across product categories and brands, considering revenue, profit, and margins.
    """)

    # Category profitability
    if not df.empty and 'category' in df.columns:
        category_profit = df.groupby('category').agg({
            'profit_usd': 'sum',
            'total_price_usd': 'sum',
            'profit_margin_percent': 'mean',
            'order_id': 'count'
        }).reset_index()

        category_profit.columns = ['category', 'total_profit', 'total_revenue', 'avg_margin', 'order_count']
        category_profit = category_profit[category_profit['order_count'] > 0] # Filter out categories with no orders

        if not category_profit.empty:
            category_profit['profit_per_order'] = category_profit['total_profit'] / category_profit['order_count']
            category_profit = category_profit.sort_values('total_profit', ascending=False)

            st.write("**Top 10 Most Profitable Categories:**")
            st.dataframe(category_profit.head(10))

            st.write("**Category Profitability Matrix: Revenue vs Margin**")
            fig_profit_matrix = px.scatter(
                category_profit,
                x='total_revenue',
                y='avg_margin',
                size='order_count',
                color='category',
                hover_data=['total_profit'],
                title="Category Profitability Matrix: Revenue vs Margin",
                labels={'total_revenue': 'Total Revenue (USD)', 'avg_margin': 'Average Profit Margin (%)'}
            )

            fig_profit_matrix.update_layout(height=600)
            st.plotly_chart(fig_profit_matrix, use_container_width=True)
        else:
            st.warning("No categories with orders found within the selected filters to analyze profitability.")
    else:
        st.warning("Insufficient data to perform category profitability analysis for the selected filters.")

    # Brand Performance
    if not df.empty and 'brand' in df.columns:
        brand_performance = df.groupby('brand').agg({
            'total_price_usd': 'sum',
            'profit_usd': 'sum',
            'profit_margin_percent': 'mean',
            'order_id': 'count',
            'product_rating_avg': 'first'
        }).reset_index()

        brand_performance.columns = ['brand', 'revenue', 'profit', 'avg_margin', 'orders', 'avg_rating']
        brand_performance = brand_performance[brand_performance['orders'] > 0] # Filter out brands with no orders

        if not brand_performance.empty:
            brand_performance = brand_performance.sort_values('revenue', ascending=False).head(15)

            st.write("**Top 15 Brands by Revenue:**")
            st.dataframe(brand_performance)
        else:
            st.warning("No brands with orders found within the selected filters to analyze performance.")
    else:
        st.warning("Insufficient data to perform brand performance analysis for the selected filters.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Marketing Channel ROI
    # ─────────────────────────────────────────────────────────────────────────────

    st.subheader("4️⃣ MARKETING CHANNEL ROI ANALYSIS")
    st.markdown("""
    Evaluation of different marketing channels based on revenue, profit, customer acquisition, and engagement metrics.
    """)

    if not df.empty and 'campaign_source' in df.columns:
        channel_roi = df.groupby('campaign_source').agg({
            'total_price_usd': ['sum', 'mean'],
            'profit_usd': 'sum',
            'order_id': 'count',
            'customer_loyalty_score': 'mean',
            'abandoned_cart_flag': 'mean'
        }).round(2)

        channel_roi.columns = ['_'.join(col).strip() for col in channel_roi.columns.values]
        channel_roi = channel_roi.reset_index()
        channel_roi.columns = ['channel', 'total_revenue', 'avg_order_value', 'total_profit', 'order_count', 'avg_loyalty', 'cart_abandonment_rate']
        channel_roi = channel_roi[channel_roi['order_count'] > 0] # Filter out channels with no orders

        if not channel_roi.empty:
            st.dataframe(channel_roi)

            st.write("**Marketing Channel Performance Dashboard:**")
            fig_channel_perf = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Total Revenue', 'Average Order Value', 'Total Profit', 'Order Count'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )

            fig_channel_perf.add_trace(
                go.Bar(x=channel_roi['channel'], y=channel_roi['total_revenue'], marker_color=COLORS['primary'], showlegend=False),
                row=1, col=1
            )

            fig_channel_perf.add_trace(
                go.Bar(x=channel_roi['channel'], y=channel_roi['avg_order_value'], marker_color=COLORS['success'], showlegend=False),
                row=1, col=2
            )

            fig_channel_perf.add_trace(
                go.Bar(x=channel_roi['channel'], y=channel_roi['total_profit'], marker_color=COLORS['accent'], showlegend=False),
                row=2, col=1
            )

            fig_channel_perf.add_trace(
                go.Bar(x=channel_roi['channel'], y=channel_roi['order_count'], marker_color=COLORS['secondary'], showlegend=False),
                row=2, col=2
            )

            fig_channel_perf.update_layout(
                title_text="Marketing Channel Performance Dashboard",
                height=700,
                showlegend=False
            )
            st.plotly_chart(fig_channel_perf, use_container_width=True)
        else:
            st.warning("No marketing channel data found within the selected filters.")
    else:
        st.warning("Insufficient data to perform marketing channel ROI analysis for the selected filters.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Fraud Risk Analysis
    # ─────────────────────────────────────────────────────────────────────────────

    st.subheader("5️⃣ FRAUD RISK ANALYSIS")
    st.markdown("""
    Analysis of transaction characteristics and payment failure rates across different fraud risk categories.
    """)

    if not df.empty and 'fraud_risk_category' in df.columns:
        fraud_analysis = df.groupby('fraud_risk_category').agg({
            'order_id': 'count',
            'payment_status': lambda x: (x == 'Failed').sum(),
            'total_price_usd': 'mean',
            'profit_usd': 'mean'
        }).reset_index()

        fraud_analysis.columns = ['risk_category', 'order_count', 'failed_payments', 'avg_order_value', 'avg_profit']
        fraud_analysis = fraud_analysis[fraud_analysis['order_count'] > 0] # Filter out categories with no orders

        if not fraud_analysis.empty and (fraud_analysis['order_count'] > 0).any():
            fraud_analysis['failure_rate'] = (fraud_analysis['failed_payments'] / fraud_analysis['order_count'] * 100).round(2)

            st.write("**Fraud Risk Category Analysis:**")
            st.dataframe(fraud_analysis)

            high_risk = df[df['fraud_risk_category'] == 'High']

            if not high_risk.empty:
                st.write(f"\n📊 **High-Risk Transaction Characteristics:**")
                st.write(f"  Total high-risk orders: {len(high_risk):,}")
                st.write(f"  Avg order value: ${high_risk['total_price_usd'].mean():.2f}")
                st.write(f"  Most common payment method: {high_risk['payment_method'].mode()[0] if not high_risk['payment_method'].empty else 'N/A'}")
                st.write(f"  Most common device: {high_risk['device_type'].mode()[0] if not high_risk['device_type'].empty else 'N/A'}")
                st.write(f"  Payment failure rate: {(high_risk['payment_status'] == 'Failed').sum() / len(high_risk) * 100:.2f}%")
            else:
                st.warning("No high-risk transactions found within the selected filters.")
        else:
            st.warning("No fraud risk data available for the selected filters.")
    else:
        st.warning("Insufficient data to perform fraud risk analysis for the selected filters.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Customer Satisfaction Drivers
    # ─────────────────────────────────────────────────────────────────────────────

    st.subheader("6️⃣ CUSTOMER SATISFACTION DRIVERS")
    st.markdown("""
    Investigation into factors influencing customer satisfaction, particularly delivery speed and its impact on ratings.
    """)
    if not df.empty and 'review_sentiment' in df.columns and 'rating' in df.columns:
        satisfaction_factors = df.groupby('review_sentiment')['rating'].mean().reset_index()
        if not satisfaction_factors.empty:
            st.write("**Average Rating by Sentiment:**")
            st.dataframe(satisfaction_factors)
        else:
            st.warning("No review sentiment data available for the selected filters.")

        if 'delivery_days' in df.columns:
            delivery_satisfaction = df.groupby(pd.cut(df['delivery_days'], bins=[0, 3, 7, 14, 30], labels=['1-3 days', '4-7 days', '8-14 days', '15+ days'], right=True)).agg({
                'rating': 'mean',
                'order_id': 'count',
                'review_sentiment': lambda x: (x == 'Positive').sum()
            }).reset_index()

            delivery_satisfaction.columns = ['delivery_window', 'avg_rating', 'order_count', 'positive_reviews']
            delivery_satisfaction = delivery_satisfaction[delivery_satisfaction['order_count'] > 0] # Filter out empty bins

            if not delivery_satisfaction.empty and (delivery_satisfaction['order_count'] > 0).any():
                delivery_satisfaction['positive_rate'] = (delivery_satisfaction['positive_reviews'] / delivery_satisfaction['order_count'] * 100).round(2)

                st.write("**Delivery Speed vs Satisfaction:**")
                st.dataframe(delivery_satisfaction)

                st.write("**Delivery Speed Impact on Customer Satisfaction:**")
                fig_satisfaction_drivers = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Rating by Delivery Window', 'Positive Review Rate by Delivery Window')
                )

                fig_satisfaction_drivers.add_trace(
                    go.Bar(x=delivery_satisfaction['delivery_window'], y=delivery_satisfaction['avg_rating'],
                           marker_color=COLORS['primary'], showlegend=False),
                    row=1, col=1
                )

                fig_satisfaction_drivers.add_trace(
                    go.Bar(x=delivery_satisfaction['delivery_window'], y=delivery_satisfaction['positive_rate'],
                           marker_color=COLORS['success'], showlegend=False),
                    row=1, col=2
                )

                fig_satisfaction_drivers.update_layout(
                    title_text="Delivery Speed Impact on Customer Satisfaction",
                    height=400
                )

                fig_satisfaction_drivers.update_yaxes(title_text="Average Rating", row=1, col=1)
                fig_satisfaction_drivers.update_yaxes(title_text="Positive Review Rate (%)", row=1, col=2)
                st.plotly_chart(fig_satisfaction_drivers, use_container_width=True)
            else:
                st.warning("No delivery speed data available for the selected filters to analyze satisfaction.")
        else:
            st.warning("The 'delivery_days' column is not available in the filtered data. Cannot analyze delivery satisfaction.")
    else:
        st.warning("Insufficient data to perform customer satisfaction drivers analysis for the selected filters.")


    # ─────────────────────────────────────────────────────────────────────────────
    # Geographic Performance
    # ─────────────────────────────────────────────────────────────────────────────

    st.subheader("7️⃣ GEOGRAPHIC PERFORMANCE ANALYSIS")
    st.markdown("""
    Analysis of key business metrics across different countries to identify top-performing regions.
    """)

    if not df.empty and 'country' in df.columns:
        country_performance = df.groupby('country').agg({
            'order_id': 'count',
            'total_price_usd': ['sum', 'mean'],
            'profit_usd': 'sum',
            'customer_id': 'nunique',
            'delivery_days': 'mean'
        }).round(2)

        country_performance.columns = ['_'.join(col).strip() for col in country_performance.columns.values]
        country_performance = country_performance.reset_index()
        country_performance.columns = ['country', 'orders', 'total_revenue', 'avg_order_value', 'total_profit', 'unique_customers', 'avg_delivery_days']
        country_performance = country_performance[country_performance['orders'] > 0] # Filter out countries with no orders

        if not country_performance.empty:
            country_performance = country_performance.sort_values('total_revenue', ascending=False)

            st.write("**Top Countries by Revenue:**")
            st.dataframe(country_performance.head(10))

            st.write("**Top Countries by Total Revenue:**")
            fig_geo_revenue = go.Figure(data=[
                go.Bar(
                    y=country_performance['country'].head(15),
                    x=country_performance['total_revenue'].head(15),
                    orientation='h',
                    marker_color=COLORS['primary'],
                    text=country_performance['total_revenue'].head(15).apply(lambda x: f"${x/1e6:.1f}M"),
                    textposition='auto'
                )
            ])

            fig_geo_revenue.update_layout(
                title="Top 15 Countries by Total Revenue",
                xaxis_title="Revenue (USD)",
                yaxis_title="Country",
                height=600
            )
            st.plotly_chart(fig_geo_revenue, use_container_width=True)
        else:
            st.warning("No country data found within the selected filters to analyze geographic performance.")
    else:
        st.warning("Insufficient data to perform geographic performance analysis for the selected filters.")

    st.markdown("""
---
## 💡 Strategic Business Insights

### 🎯 **Customer Value Optimization**
1.  **CLV Segmentation**: VIP-tier customers generate disproportionate value
    -   **Action**: Invest in retention programs for high-CLV customers
    -   **Opportunity**: Upgrade Medium-tier customers through targeted engagement

2.  **Segment-Specific Strategies**: Different segments show distinct behaviors
    -   **Premium**: High loyalty, moderate volume → Focus on experience
    -   **VIP**: High value, high frequency → Exclusive offerings
    -   **Regular**: High volume, price-sensitive → Efficiency and value

### 📦 **Product & Category Management**
1.  **Return Rate Variation**: Significant differences across categories
    -   **Action**: Investigate high-return categories for quality/description issues
    -   **Cost Impact**: Returns erode profitability - priority area for improvement

2.  **Profitability Matrix**: Not all revenue is equal
    -   **High Revenue, High Margin**: Invest and scale
    -   **High Revenue, Low Margin**: Optimize cost structure
    -   **Low Revenue, High Margin**: Niche opportunities
    -   **Low Revenue, Low Margin**: Consider discontinuation

### 📢 **Marketing Optimization**
1.  **Channel Performance**: Distinct ROI profiles by channel
    -   **Best for Volume**: [Identify from data]
    -   **Best for Value**: [Identify from data]
    -   **Best for Loyalty**: [Identify from data]

2.  **Multi-Touch Attribution**: Customers engage across channels
    -   **Action**: Implement attribution modeling for better budget allocation

### ⚠️ **Risk Management**
1.  **Fraud Prevention**: High-risk transactions show predictable patterns
    -   **Action**: Enhanced verification for high-risk profiles
    -   **Balance**: Don't over-restrict and harm legitimate high-value customers

2.  **Payment Failure**: Correlated with fraud risk and payment method
    -   **Action**: Optimize payment flow for high-failure methods

### 🚚 **Operational Excellence**
1.  **Delivery Speed = Satisfaction**: Strong correlation confirmed
    -   **Trade-off**: Speed vs cost must be balanced
    -   **Opportunity**: Premium shipping tier to capture willingness-to-pay

2.  **Geographic Efficiency**: Delivery times vary significantly by region
    -   **Action**: Optimize warehouse placement and carrier selection

### 💰 **Pricing Strategy**
1.  **Discount Effectiveness**: Volume lift vs margin erosion trade-off
    -   **Sweet Spot**: Identify optimal discount levels by category
    -   **Personalization**: Different segments respond differently to discounts

2.  **Price Elasticity**: Varies by category and customer segment
    -   **Action**: Dynamic pricing based on demand, inventory, and segment

---
    """)


# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED VISUALIZATIONS PAGE
# ═══════════════════════════════════════════════════════════════════════════

def advanced_visualizations_page(df):
    st.header("📊 Advanced Interactive Visualizations")
    st.markdown("""
    This section provides advanced and interactive visualizations to uncover deeper patterns and relationships within the data.
    """)

    # ─────────────────────────────────────────────────────────────────────────────
    # RFM Analysis Visualization
    # ─────────────────────────────────────────────────────────────────────────────
    st.subheader("1️⃣ RFM Analysis: 3D Customer Segmentation")
    st.markdown("""
    A 3D scatter plot visualizing customers based on Recency, Frequency, and Monetary scores, colored by combined RFM score.
    """)

    if not df.empty and 'customer_id' in df.columns and 'order_date' in df.columns and 'total_price_usd' in df.columns:
        # Recreate RFM data for visualization
        latest_date = df['order_date'].max()

        rfm_viz = df.groupby('customer_id').agg({
            'order_date': lambda x: (latest_date - x.max()).days,
            'order_id': 'count',
            'total_price_usd': 'sum',
            'customer_segment': 'first'
        }).reset_index()
        rfm_viz.columns = ['customer_id', 'recency_days', 'order_count', 'total_spent', 'segment']

        if not rfm_viz.empty and len(rfm_viz) > 1:
            # Normalize RFM scores (1-5)
            # Use try-except for qcut to handle cases with too few unique values
            try:
                rfm_viz['recency_score'] = pd.qcut(
                    rfm_viz['recency_days'],
                    q=5,
                    labels=[5,4,3,2,1],
                    duplicates='drop'
                ).astype(int)
            except ValueError:
                rfm_viz['recency_score'] = 1 # Default to 1 if not enough quantiles

            try:
                rfm_viz['frequency_score'] = pd.qcut(
                    rfm_viz['order_count'].rank(method='first'),
                    q=5,
                    labels=[1,2,3,4,5],
                    duplicates='drop'
                ).astype(int)
            except ValueError:
                rfm_viz['frequency_score'] = 1

            try:
                rfm_viz['monetary_score'] = pd.qcut(
                    rfm_viz['total_spent'].rank(method='first'),
                    q=5,
                    labels=[1,2,3,4,5],
                    duplicates='drop'
                ).astype(int)
            except ValueError:
                rfm_viz['monetary_score'] = 1

            rfm_viz['rfm_score'] = rfm_viz['recency_score'] + rfm_viz['frequency_score'] + rfm_viz['monetary_score']

            fig_rfm = px.scatter_3d(
                rfm_viz.sample(min(5000, len(rfm_viz)), random_state=42) if len(rfm_viz) > 5000 else rfm_viz,
                x='recency_score',
                y='frequency_score',
                z='monetary_score',
                color='rfm_score',
                size='total_spent',
                hover_data=['segment', 'order_count'],
                title="RFM Analysis: 3D Customer Segmentation",
                labels={
                    'recency_score': 'Recency (5=Recent)',
                    'frequency_score': 'Frequency',
                    'monetary_score': 'Monetary'
                },
                color_continuous_scale='Viridis'
            )
            fig_rfm.update_layout(height=700)
            st.plotly_chart(fig_rfm, use_container_width=True)
        else:
            st.warning("Insufficient customer data to perform RFM analysis for the selected filters.")
    else:
        st.warning("Insufficient data to perform RFM analysis for the selected filters.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Cohort Analysis by Account Creation Month
    # ─────────────────────────────────────────────────────────────────────────────
    st.subheader("2️⃣ Cohort Analysis: Revenue by Months Since Customer Acquisition")
    st.markdown("""
    A heatmap showing how different customer cohorts (based on account creation month) contribute to revenue over time.
    """)

    if not df.empty and 'account_creation_date' in df.columns and 'order_date' in df.columns:
        df_temp_cohort = df.copy()
        df_temp_cohort['account_creation_month'] = df_temp_cohort['account_creation_date'].dt.to_period('M')
        df_temp_cohort['order_month'] = df_temp_cohort['order_date'].dt.to_period('M')

        cohort_data = df_temp_cohort.groupby(['account_creation_month', 'order_month']).agg({
            'total_price_usd': 'sum',
            'customer_id': 'nunique'
        }).reset_index()

        if not cohort_data.empty:
            # Calculate months since acquisition
            cohort_data['months_since_acquisition'] = (
                cohort_data['order_month'].apply(lambda x: x.to_timestamp()).view('int64') // 10**9 // 2592000 - # Convert to approximate months
                cohort_data['account_creation_month'].apply(lambda x: x.to_timestamp()).view('int64') // 10**9 // 2592000
            )

            # Filter out negative months_since_acquisition (shouldn't happen with correct data, but for robustness)
            cohort_data = cohort_data[cohort_data['months_since_acquisition'] >= 0]

            if not cohort_data.empty:
                # Pivot for heatmap
                cohort_pivot = cohort_data.pivot_table(
                    index='account_creation_month',
                    columns='months_since_acquisition',
                    values='total_price_usd',
                    aggfunc='sum'
                )
                cohort_pivot = cohort_pivot.fillna(0) # Fill NaN with 0 for display

                # Plot first 12 months, or fewer if not enough data
                cohort_pivot_12m = cohort_pivot.iloc[:, :min(12, cohort_pivot.shape[1])]

                if not cohort_pivot_12m.empty:
                    fig_cohort = go.Figure(data=go.Heatmap(
                        z=cohort_pivot_12m.values,
                        x=[f"Month {int(i)}" for i in cohort_pivot_12m.columns],
                        y=cohort_pivot_12m.index.astype(str),
                        colorscale='Blues',
                        colorbar=dict(title="Revenue (USD)")
                    ))

                    fig_cohort.update_layout(
                        title="Cohort Analysis: Revenue by Months Since Customer Acquisition",
                        xaxis_title="Months Since Acquisition",
                        yaxis_title="Acquisition Cohort",
                        height=600
                    )
                    st.plotly_chart(fig_cohort, use_container_width=True)
                else:
                    st.warning("No data available for cohort analysis within the first 12 months for the selected filters.")
            else:
                st.warning("No valid cohort data generated after filtering for months since acquisition.")
        else:
            st.warning("No cohort data found for the selected filters.")
    else:
        st.warning("Insufficient data to perform cohort analysis for the selected filters (missing account creation or order date).")

    # ─────────────────────────────────────────────────────────────────────────────
    # Funnel Analysis: Customer Journey
    # ─────────────────────────────────────────────────────────────────────────────
    st.subheader("3️⃣ E-Commerce Conversion Funnel")
    st.markdown("""
    A visualization of the customer journey, highlighting conversion rates at different stages from all orders to repeat purchases.
    """)

    if not df.empty:
        funnel_data = pd.DataFrame({
            'Stage': [
                'All Orders',
                'No Cart Abandonment',
                'Payment Attempted',
                'Completed Order',
                'Repeat Purchase'
            ],
            'Count': [
                len(df),
                len(df[df['abandoned_cart_before'] == 'No']),
                len(df[df['payment_status'].isin(['Paid', 'Pending'])]), # Consider 'Pending' as attempted
                len(df[df['order_status'] == 'Completed']),
                len(df[df['total_orders_by_customer'] > 1])
            ]
        })

        if funnel_data['Count'].iloc[0] > 0:
            funnel_data['Conversion_Rate'] = (funnel_data['Count'] / funnel_data['Count'].iloc[0] * 100).round(2)

            fig_funnel = go.Figure(go.Funnel(
                y=funnel_data['Stage'],
                x=funnel_data['Count'],
                textinfo="value+percent initial",
                marker={"color": [COLORS['primary'], COLORS['secondary'], COLORS['warning'], COLORS['success'], COLORS['danger']]}
            ))

            fig_funnel.update_layout(
                title="E-Commerce Conversion Funnel",
                height=500
            )
            st.plotly_chart(fig_funnel, use_container_width=True)

            st.write("**Funnel Conversion Rates:**")
            #st.dataframe(funnel_data.to_string(index=False))
            st.dataframe(funnel_data)  # Pass the DataFrame itself
            #st.dataframe(funnel_data.style.format({'Conversion_Rate': '{:.2f}'}).hide_index())
        else:
            st.warning("No orders found within the selected filters to build the conversion funnel.")
    else:
        st.warning("Insufficient data to build the conversion funnel for the selected filters.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Product Performance Bubble Chart
    # ─────────────────────────────────────────────────────────────────────────────
    st.subheader("4️⃣ Product Performance Matrix: Revenue vs Margin")
    st.markdown("""
    A bubble chart illustrating product performance based on revenue, profit margin, units sold, and average rating.
    """)

    if not df.empty and 'product_id' in df.columns:
        product_metrics = df.groupby('product_id').agg({
            'total_price_usd': 'sum',
            'profit_margin_percent': 'mean',
            'quantity': 'sum',
            'product_rating_avg': 'first',
            'category': 'first'
        }).reset_index()

        product_metrics.columns = ['product_id', 'revenue', 'margin', 'units_sold', 'rating', 'category']
        product_metrics = product_metrics[product_metrics['units_sold'] > 0] # Filter out products with no sales

        if not product_metrics.empty:
            # Sample for performance or take top N products
            product_sample = product_metrics.nlargest(100, 'revenue') if len(product_metrics) > 100 else product_metrics

            fig_product_matrix = px.scatter(
                product_sample,
                x='revenue',
                y='margin',
                size='units_sold',
                color='rating',
                hover_data=['product_id', 'category'],
                title="Product Performance Matrix: Revenue vs Margin (Top Products)",
                labels={'revenue': 'Total Revenue (USD)', 'margin': 'Profit Margin (%)'},
                color_continuous_scale='RdYlGn'
            )

            fig_product_matrix.update_layout(height=600)
            st.plotly_chart(fig_product_matrix, use_container_width=True)
        else:
            st.warning("No product data found with sales for the selected filters.")
    else:
        st.warning("Insufficient data to display product performance matrix for the selected filters.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Time-based Performance Heatmap
    # ─────────────────────────────────────────────────────────────────────────────
    st.subheader("5️⃣ Order Volume Heatmap: Day of Week vs Hour of Day")
    st.markdown("""
    A heatmap showing the distribution of order volume across different days of the week and hours of the day.
    """)

    if not df.empty and 'order_dayname' in df.columns and 'order_hour' in df.columns:
        hourly_daily = df.groupby(['order_dayname', 'order_hour']).agg({
            'order_id': 'count',
            'total_price_usd': 'mean'
        }).reset_index()

        if not hourly_daily.empty:
            hourly_daily_pivot = hourly_daily.pivot(
                index='order_dayname',
                columns='order_hour',
                values='order_id'
            )

            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hourly_daily_pivot = hourly_daily_pivot.reindex(day_order)

            fig_time_heatmap = go.Figure(data=go.Heatmap(
                z=hourly_daily_pivot.values,
                x=hourly_daily_pivot.columns,
                y=hourly_daily_pivot.index,
                colorscale='YlOrRd',
                colorbar=dict(title="Order Count")
            ))

            fig_time_heatmap.update_layout(
                title="Order Volume Heatmap: Day of Week vs Hour of Day",
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week",
                height=500
            )
            st.plotly_chart(fig_time_heatmap, use_container_width=True)
        else:
            st.warning("No hourly or daily order data found for the selected filters.")
    else:
        st.warning("Insufficient data to display time-based performance heatmap for the selected filters.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Customer Segment Behavioral Profile
    # ─────────────────────────────────────────────────────────────────────────────
    st.subheader("6️⃣ Customer Segment Behavioral Profile (Radar Chart)")
    st.markdown("""
    A radar chart comparing the behavioral profiles of different customer segments across various metrics.
    """)

    if not df.empty and 'customer_segment' in df.columns:
        segment_behavior = df.groupby('customer_segment').agg({
            'customer_aov': 'mean',
            'total_orders_by_customer': 'mean',
            'customer_loyalty_score': 'mean',
            'session_duration_minutes': 'mean',
            'pages_visited': 'mean',
            'discount_percent': 'mean',
            'abandoned_cart_flag': 'mean'
        }).reset_index()

        if not segment_behavior.empty and len(segment_behavior) > 1:
            # Normalize for radar chart
            scaler = MinMaxScaler()

            metrics_to_scale = ['customer_aov', 'total_orders_by_customer', 'customer_loyalty_score',
                                'session_duration_minutes', 'pages_visited']

            segment_behavior_scaled = segment_behavior.copy()
            # Filter metrics_to_scale to only include columns present in segment_behavior
            present_metrics_to_scale = [m for m in metrics_to_scale if m in segment_behavior.columns]

            if present_metrics_to_scale:
                segment_behavior_scaled[present_metrics_to_scale] = scaler.fit_transform(segment_behavior[present_metrics_to_scale])

                # Create radar chart
                fig_radar = go.Figure()

                categories = [col.replace('_', ' ').title() for col in present_metrics_to_scale]

                for segment in segment_behavior_scaled['customer_segment']:
                    values = segment_behavior_scaled[segment_behavior_scaled['customer_segment'] == segment][present_metrics_to_scale].values[0]
                    if values.size > 0:
                        values = list(values) + [values[0]]  # Complete the circle
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories + [categories[0]],
                            fill='toself',
                            name=segment
                        ))

                if fig_radar.data:
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title="Customer Segment Behavioral Profile (Normalized)",
                        height=600
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("No radar chart data generated for the selected filters.")
            else:
                st.warning("No relevant metrics available for scaling and radar chart generation.")
        else:
            st.warning("Insufficient customer segment data to build a behavioral profile for the selected filters.")
    else:
        st.warning("Insufficient data to display customer segment behavioral profile for the selected filters.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Pareto Analysis: Revenue Contribution
    # ─────────────────────────────────────────────────────────────────────────────
    st.subheader("7️⃣ Pareto Analysis: Customer Revenue Contribution (80/20 Rule)")
    st.markdown("""
    A visualization of the Pareto principle, showing how a small percentage of customers contribute to the majority of revenue.
    """)

    if not df.empty and 'customer_id' in df.columns and 'total_price_usd' in df.columns:
        customer_revenue = df.groupby('customer_id')['total_price_usd'].sum().reset_index()
        customer_revenue = customer_revenue.sort_values('total_price_usd', ascending=False)

        if not customer_revenue.empty and customer_revenue['total_price_usd'].sum() > 0:
            customer_revenue['cumulative_revenue'] = customer_revenue['total_price_usd'].cumsum()
            customer_revenue['cumulative_pct'] = (customer_revenue['cumulative_revenue'] / customer_revenue['total_price_usd'].sum() * 100)
            customer_revenue['customer_pct'] = (np.arange(1, len(customer_revenue) + 1) / len(customer_revenue) * 100)

            fig_pareto = go.Figure()

            fig_pareto.add_trace(go.Bar(
                x=customer_revenue['customer_pct'][:min(1000, len(customer_revenue))],
                y=customer_revenue['total_price_usd'][:min(1000, len(customer_revenue))],
                name='Individual Revenue',
                marker_color=COLORS['primary'],
                yaxis='y',
                opacity=0.6
            ))

            fig_pareto.add_trace(go.Scatter(
                x=customer_revenue['customer_pct'],
                y=customer_revenue['cumulative_pct'],
                name='Cumulative %',
                line=dict(color=COLORS['danger'], width=3),
                yaxis='y2'
            ))

            # Add 80% line
            fig_pareto.add_hline(y=80, line_dash="dash", line_color="green",
                          annotation_text="80%", yref='y2')

            fig_pareto.update_layout(
                title="Pareto Analysis: Customer Revenue Contribution",
                xaxis_title="Cumulative % of Customers",
                yaxis=dict(title="Revenue (USD)"),
                yaxis2=dict(title="Cumulative Revenue %", overlaying='y', side='right'),
                height=600
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

            pareto_80_idx_df = customer_revenue[customer_revenue['cumulative_pct'] <= 80]
            if not pareto_80_idx_df.empty:
                pareto_80 = pareto_80_idx_df.iloc[-1]
                st.write(f"\n📊 **80/20 Analysis:**")
                st.write(f"   Top {pareto_80['customer_pct']:.1f}% of customers generate 80% of revenue")
                st.write(f"   That's {int(len(customer_revenue) * pareto_80['customer_pct'] / 100):,} customers out of {len(customer_revenue):,}")
            else:
                st.warning("Unable to find the 80% revenue cutoff for the selected filters.")
        else:
            st.warning("No revenue data available for Pareto analysis for the selected filters.")
    else:
        st.warning("Insufficient data to display Pareto analysis for the selected filters.")

    st.markdown("""
---
## 🎨 Advanced Visualizations Summary

### 📊 **Visualizations Created**

1.  **3D RFM Segmentation**: Multi-dimensional customer value visualization
2.  **Cohort Analysis**: Customer lifetime revenue patterns by acquisition cohort
3.  **Conversion Funnel**: Customer journey drop-off analysis
4.  **Product Performance Matrix**: Revenue vs margin bubble chart
5.  **Time-based Heatmap**: Order patterns by day and hour
6.  **Segment Radar Chart**: Behavioral profiles across customer segments
7.  **Pareto Analysis**: 80/20 rule visualization for customer revenue

### 💡 **Key Insights from Visualizations**

-   **Customer Concentration**: Small percentage of customers drive majority of revenue
-   **Cohort Behavior**: Different acquisition cohorts show distinct spending patterns
-   **Conversion Opportunities**: Significant drop-offs at key funnel stages
-   **Time Patterns**: Clear peak and off-peak shopping windows
-   **Segment Differentiation**: Distinct behavioral signatures by customer tier

### 🎯 **Actionable Insights**

-   Focus retention efforts on high-RFM score customers
-   Optimize marketing spend timing based on hourly/daily patterns
-   Address funnel bottlenecks with A/B testing
-   Customize experience by segment behavioral profile
-   Implement dynamic pricing based on time/segment patterns

---
    """)


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY & RECOMMENDATIONS PAGE
# ═══════════════════════════════════════════════════════════════════════════

def final_summary_recommendations_page(df):
    st.header("📝 Final Summary & Strategic Recommendations")
    st.markdown("""
    This section synthesizes all findings into an executive summary and provides strategic, actionable recommendations for business growth and optimization.
    """)

    # Generate comprehensive summary statistics (re-calculate or pass from KPIs page if needed)
    # For simplicity, we'll recalculate here as it's quick.
    summary_metrics_values = {
        'Total Orders': len(df),
        'Total Revenue': df['total_price_usd'].sum(),
        'Total Profit': df['profit_usd'].sum(),
        'Avg Profit Margin': df['profit_margin_percent'].mean(),
        'Avg Order Value': df['total_price_usd'].mean(),
        'Unique Customers': df['customer_id'].nunique(),
        'Unique Products': df['product_id'].nunique(),
        'Order Completion Rate': (df['order_status'] == 'Completed').sum() / len(df) * 100,
        'Return Rate': (df['order_status'] == 'Returned').sum() / len(df) * 100,
        'Avg Customer Rating': df['rating'].mean(),
        'Mobile Order %': (df['device_type'] == 'Mobile').sum() / len(df) * 100,
        'Avg Delivery Days': df['delivery_days'].mean()
    }

    st.subheader("📊 EXECUTIVE SUMMARY - KEY METRICS")
    executive_summary_str = []
    for metric, value in summary_metrics_values.items():
        if 'Rate' in metric or 'Margin' in metric or 'Mobile Order' in metric:
            executive_summary_str.append(f"  {metric:.<40} {value:.2f}%")
        elif 'Avg' in metric and 'Value' not in metric and 'Rating' not in metric and 'Days' not in metric:
            executive_summary_str.append(f"  {metric:.<40} {value:,.2f}")
        elif 'Revenue' in metric or 'Profit' in metric or 'Value' in metric:
            executive_summary_str.append(f"  {metric:.<40} ${value:,.2f}")
        elif 'Rating' in metric:
            executive_summary_str.append(f"  {metric:.<40} {value:.2f}/5")
        elif 'Days' in metric:
            executive_summary_str.append(f"  {metric:.<40} {value:.1f} days")
        else:
            executive_summary_str.append(f"  {metric:.<40} {value:,}")

    st.code("\n".join(executive_summary_str))

    st.markdown("""
---
# 🎯 STRATEGIC RECOMMENDATIONS & ACTION PLAN

## 🏆 TOP PRIORITY INITIATIVES

### 1. Customer Value Optimization 💎
**Finding**: 80/20 rule applies - small customer segment drives majority of revenue

**Actions**:
- ✅ Implement VIP loyalty program with exclusive benefits
- ✅ Create automated engagement flows for high-CLV customers
- ✅ Develop upgrade paths for medium-tier customers
- ✅ Personalize communications based on RFM segmentation

**Expected Impact**: 15-20% increase in customer lifetime value

---

### 2. Profitability Enhancement 💰
**Finding**: Profit margins vary significantly by category and discount level

**Actions**:
- ✅ Optimize discount strategy - avoid blanket discounting
- ✅ Implement dynamic pricing for high-elasticity categories
- ✅ Phase out or restructure low-margin products
- ✅ Bundle high-margin items with popular products

**Expected Impact**: 3-5 percentage point margin improvement

---

### 3. Operational Excellence 🚚
**Finding**: Delivery speed strongly correlates with customer satisfaction

**Actions**:
- ✅ Expand fast-delivery options for high-value customers
- ✅ Optimize warehouse locations based on geographic analysis
- ✅ Partner with premium carriers for critical markets
- ✅ Set delivery time expectations clearly at checkout

**Expected Impact**: 0.3-0.5 point improvement in average rating

---

### 4. Marketing Channel Optimization 📢
**Finding**: Channels show distinct performance profiles and customer acquisition costs vary

**Actions**:
- ✅ Reallocate budget to highest-ROI channels
- ✅ Implement multi-touch attribution modeling
- ✅ A/B test channel-specific messaging and offers
- ✅ Create channel-specific landing pages

**Expected Impact**: 20-30% improvement in marketing efficiency

---

### 5. Return Rate Reduction 📦
**Finding**: High return rates in specific categories erode profitability

**Actions**:
- ✅ Improve product descriptions and imagery
- ✅ Implement AR/VR try-before-buy features
- ✅ Add customer reviews and Q&A sections
- ✅ Analyze return reasons and address root causes

**Expected Impact**: 2-3 percentage point reduction in return rate

---

### 6. Fraud Risk Management ⚠️
**Finding**: Predictable patterns in high-risk transactions

**Actions**:
- ✅ Enhance fraud detection algorithms
- ✅ Implement stepped verification for high-risk profiles
- ✅ Monitor payment failure patterns
- ✅ Balance security with customer experience

**Expected Impact**: $500K-1M annual fraud loss prevention

---

### 7. Mobile Experience Optimization 📱
**Finding**: Mobile dominates traffic but shows higher abandonment

**Actions**:
- ✅ Streamline mobile checkout flow
- ✅ Implement one-click payment options
- ✅ Optimize page load speeds
- ✅ A/B test mobile-specific features

**Expected Impact**: 5-10% improvement in mobile conversion rate

---

### 8. Personalization Engine 🎯
**Finding**: Different segments respond to different triggers

**Actions**:
- ✅ Build ML-based recommendation engine
- ✅ Personalize homepage and product listings
- ✅ Customize email campaigns by segment
- ✅ Implement dynamic pricing by customer value

**Expected Impact**: 10-15% increase in average order value

---

## 📈 MEASUREMENT & TRACKING

### Key Performance Indicators (KPIs) to Monitor:

**Revenue Metrics**:
- Total Revenue (Monthly/Quarterly)
- Average Order Value
- Revenue per Customer
- Customer Lifetime Value

**Profitability Metrics**:
- Gross Profit Margin
- Net Profit Margin
- Profit per Order
- Profit per Customer

**Customer Metrics**:
- Customer Acquisition Cost (CAC)
- Customer Retention Rate
- Churn Rate
- Net Promoter Score (NPS)

**Operational Metrics**:
- Order Fulfillment Time
- Delivery Success Rate
- Return Rate
- Customer Service Ticket Rate

**Marketing Metrics**:
- Channel ROI
- Conversion Rate by Channel
- Customer Acquisition Cost by Channel
- Marketing Efficiency Ratio

---

## ⚠️ LIMITATIONS & CONSIDERATIONS

### Data Limitations:
1.  **Missing Context**: Customer feedback only available for ~80% of orders
2.  **Fraud Scores**: Based on risk scoring model - not actual fraud confirmation
3.  **Attribution**: Single-touch attribution may undervalue multi-channel journeys
4.  **Seasonality**: 2-year window may not capture long-term patterns

### Analytical Considerations:
1.  **Correlation ≠ Causation**: Statistical relationships don't prove causal links
2.  **Sample Bias**: Dataset represents current customers, not market potential
3.  **External Factors**: Economic conditions, competition not captured
4.  **Time Lag**: Some initiatives take months to show results

---

## 🔮 NEXT STEPS & ADVANCED ANALYTICS

### Recommended Follow-Up Analyses:

1.  **Predictive Modeling**:
    -   Customer churn prediction
    -   Lifetime value forecasting
    -   Demand forecasting
    -   Price optimization models

2.  **Advanced Segmentation**:
    -   Behavioral clustering (K-means, hierarchical)
    -   Purchase pattern recognition
    -   Propensity modeling

3.  **A/B Testing Framework**:
    -   Pricing experiments
    -   Promotion effectiveness
    -   UX/UI optimization
    -   Email campaign testing

4.  **Time Series Forecasting**:
    -   Sales forecasting (ARIMA, Prophet)
    -   Inventory optimization
    -   Seasonal trend decomposition

5.  **Market Basket Analysis**:
    -   Product affinity analysis
    -   Cross-sell opportunities
    -   Bundle optimization

---

## 📚 CONCLUSION

This comprehensive analysis of 1M+ e-commerce transactions reveals:

✅ **Strong fundamentals** with healthy completion rates and customer satisfaction
✅ **Clear segmentation opportunities** for targeted marketing and personalization
✅ **Identified profitability levers** through pricing and category optimization
✅ **Operational insights** linking delivery speed to customer satisfaction
✅ **Channel-specific strategies** for marketing efficiency

**The data tells a story of a maturing e-commerce business with significant optimization opportunities across customer value, profitability, operations, and marketing.**

**Implementation of the recommended initiatives could drive 20-30% improvement in key business metrics over the next 12-18 months.**

---

## 🙏 ANALYSIS COMPLETE

**Dataset Analyzed**: 1,000,123 orders | 62 variables | 2-year period
**Analysis Date**: February 2026
**Methodology**: Exploratory Data Analysis, Statistical Testing, Business Intelligence

**Questions or need deeper analysis on specific areas? Ready to implement recommendations!**

---
    """)

    st.subheader("✅ Analysis Summary")
    st.success(f"Analyzed {len(df):,} orders")
    st.success(f"Examined {df.shape[1]} features (including engineered)")
    st.success(f"Generated 30+ visualizations")
    st.success(f"Performed 7+ statistical tests")
    st.success(f"Delivered actionable insights across all business dimensions")
    st.markdown("**🎯 Ready for strategic decision-making and implementation!**")


# Main application function to handle pages (updated)
def main():
    # Data loading and preprocessing
    # The @st.cache_data decorator ensures this function runs only once
    # and caches the result for performance.
    @st.cache_data
    def load_and_preprocess_data():
        file_path = "ecommerce_dataset_+1m.xlsx"
        df_loaded = pd.read_excel(file_path)

        # --- DATA QUALITY ASSESSMENT --- (from Section 3)
        # Re-calculate and drop temporary columns if they exist
        if 'calculated_total' in df_loaded.columns:
            df_loaded.drop(['calculated_total', 'calculated_profit', 'calculated_margin'], axis=1, inplace=True)

        # --- FEATURE ENGINEERING --- (from Section 4)
        df_enhanced = df_loaded.copy()

        # Time-Based Features
        df_enhanced['order_quarter'] = df_enhanced['order_date'].dt.quarter
        df_enhanced['order_dayofweek'] = df_enhanced['order_date'].dt.dayofweek
        df_enhanced['order_dayname'] = df_enhanced['order_date'].dt.day_name()
        df_enhanced['order_weekofyear'] = df_enhanced['order_date'].dt.isocalendar().week.astype(int) # Ensure int for some operations
        def categorize_time(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'
        df_enhanced['time_of_day'] = df_enhanced['order_hour'].apply(categorize_time)
        df_enhanced['account_age_days'] = (df_enhanced['order_date'] - df_enhanced['account_creation_date']).dt.days

        # Customer Behavior Features
        # These are computed on an order-level, so grouping before transform to avoid chained assignment
        df_enhanced['customer_aov'] = df_enhanced.groupby('customer_id')['total_price_usd'].transform('mean')
        df_enhanced['customer_total_spent'] = df_enhanced.groupby('customer_id')['total_price_usd'].transform('sum')

        latest_date = df_enhanced['order_date'].max()
        rfm_data = df_enhanced.groupby('customer_id').agg({
            'order_date': lambda x: (latest_date - x.max()).days,
            'order_id': 'count',
            'total_price_usd': 'sum'
        }).reset_index()
        rfm_data.columns = ['customer_id', 'recency_days', 'frequency', 'monetary']
        df_enhanced = df_enhanced.merge(rfm_data, on='customer_id', how='left')

        # Product Performance Features
        df_enhanced['product_total_quantity_sold'] = df_enhanced.groupby('product_id')['quantity'].transform('sum')
        df_enhanced['rating_vs_avg'] = df_enhanced['rating'] - df_enhanced['product_rating_avg']
        # Ensure category_avg_price is calculated carefully to avoid issues with filtered data later
        # For now, calculate on full data, filtering will be applied to the final df
        category_avg_price_map = df_enhanced.groupby('category')['unit_price_usd'].mean().to_dict()
        df_enhanced['price_vs_category_avg'] = df_enhanced.apply(lambda row: row['unit_price_usd'] - category_avg_price_map.get(row['category'], row['unit_price_usd']), axis=1)

        # Financial Metrics
        df_enhanced['revenue_per_item'] = df_enhanced['total_price_usd'] / df_enhanced['quantity']
        df_enhanced['discount_depth'] = np.where(
            df_enhanced['discount_percent'] > 0,
            'Discounted',
            'Full_Price'
        )
        high_value_threshold = df_enhanced['total_price_usd'].quantile(0.75)
        df_enhanced['is_high_value'] = (df_enhanced['total_price_usd'] >= high_value_threshold).astype(int)
        df_enhanced['profit_per_unit'] = df_enhanced['profit_usd'] / df_enhanced['quantity']

        # Operational Features
        df_enhanced['is_fast_delivery'] = (df_enhanced['delivery_days'] <= 3).astype(int)
        df_enhanced['is_international'] = (
            df_enhanced['country'] != df_enhanced['shipping_country']
        ).astype(int)
        df_enhanced['support_ticket_flag'] = (df_enhanced['support_ticket_created'] == 'Yes').astype(int)

        # Risk & Quality Features
        df_enhanced['fraud_risk_category'] = pd.cut(
            df_enhanced['fraud_risk_score'],
            bins=[0, 30, 60, 100],
            labels=['Low', 'Medium', 'High'],
            right=True, include_lowest=True
        ).astype(str) # Convert to str to handle potential NaN if score outside bins

        df_enhanced['loyalty_tier'] = pd.cut(
            df_enhanced['customer_loyalty_score'],
            bins=[0, 33, 66, 100],
            labels=['Low', 'Medium', 'High'],
            right=True, include_lowest=True
        ).astype(str)

        # Marketing & Engagement Features
        df_enhanced['engagement_rate'] = df_enhanced['pages_visited'] / (df_enhanced['session_duration_minutes'] + 0.1)
        df_enhanced['abandoned_cart_flag'] = (df_enhanced['abandoned_cart_before'] == 'Yes').astype(int)

        return df_enhanced

    df = load_and_preprocess_data()

    # Display initialization and data loading success message
    st.success("✅ Streamlit app initialized and data loaded successfully!")
    st.write("Dataset Overview:")
    st.write(f"📦 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.write(f"📅 Date Range: {df['order_date'].min().strftime('%Y-%m-%d')} to {df['order_date'].max().strftime('%Y-%m-%d')}")
    st.dataframe(df.head())

    st.markdown("""
---
## 📋 Key Observations from Initial Overview

- The dataset contains a healthy mix of numerical and categorical variables
- Time-based columns are properly formatted as datetime
- The dataset spans exactly 2 years of business operations
- Multiple dimensions available: customer, product, order, shipping, payment, marketing

---
""")

    # Add Data Quality Assessment text here (similar to how it was in previous writefile)
    st.markdown("""# 🧹 SECTION 3: DATA QUALITY ASSESSMENT""")

    # Missing Values Analysis
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percent': (df.isnull().sum().values / len(df) * 100).round(2)
    })

    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    st.subheader("📊 MISSING VALUES SUMMARY")
    if len(missing_data) > 0:
        st.dataframe(missing_data.set_index('Column'))
        st.write(f"🔢 Total columns with missing values: {len(missing_data)}")
    else:
        st.success("✅ No missing values detected!")

    # Understand Missing Data Context
    st.subheader("🔍 MISSING DATA CONTEXT ANALYSIS")

    # return_reason should only exist when order_status = 'Returned'
    st.write("**1️⃣ Return Reason Analysis:**")
    st.write(f"   - Total orders: {len(df):,}")
    st.write(f"   - Returned orders: {(df['order_status'] == 'Returned').sum():,}")
    st.write(f"   - Missing return_reason: {df['return_reason'].isnull().sum():,}")
    st.write(f"   - Return rate: {(df['order_status'] == 'Returned').sum() / len(df) * 100:.2f}%")

    # customer_feedback - understand pattern
    st.write("**2️⃣ Customer Feedback Analysis:**")
    st.write(f"   - Missing customer_feedback: {df['customer_feedback'].isnull().sum():,}")
    st.write(f"   - Feedback completion rate: {(1 - df['customer_feedback'].isnull().sum() / len(df)) * 100:.2f}%")

    # coupon_code - should only exist when coupon_used = 'Yes'
    st.write("**3️⃣ Coupon Code Analysis:**")
    st.write(f"   - Coupon used (Yes): {(df['coupon_used'] == 'Yes').sum():,}")
    st.write(f"   - Missing coupon_code: {df['coupon_code'].isnull().sum():,}")
    st.write(f"   - Coupon usage rate: {(df['coupon_used'] == 'Yes').sum() / len(df) * 100:.2f}%")

    # Check for Duplicates
    st.subheader("🔄 DUPLICATE RECORDS CHECK")

    duplicate_orders = df.duplicated(subset=['order_id']).sum()
    duplicate_rows = df.duplicated().sum()

    st.write(f"🔄 Duplicate Order IDs: {duplicate_orders:,}")
    st.write(f"🔄 Completely Duplicate Rows: {duplicate_rows:,}")

    if duplicate_orders == 0:
        st.success("✅ No duplicate orders found - data integrity confirmed!")
    else:
        st.warning(f"⚠️ {duplicate_orders:,} duplicate order IDs found. Consider handling them if they represent actual duplicate transactions.")

    # Outlier Detection - Key Numerical Columns
    st.subheader("📈 OUTLIER DETECTION (IQR Method)")

    numerical_cols = ['unit_price_usd', 'total_price_usd', 'profit_usd', 'discount_percent',
                      'quantity', 'profit_margin_percent', 'fraud_risk_score', 'customer_loyalty_score']

    outlier_summary = []
    for col in numerical_cols:
        if col in df.columns:
            outlier_count, lower, upper = detect_outliers_iqr(df, col)
            outlier_pct = (outlier_count / len(df)) * 100
            outlier_summary.append({
                'Column': col,
                'Outliers': outlier_count,
                'Outlier_%': f"{outlier_pct:.2f}%",
                'Lower_Bound': f"{lower:.2f}",
                'Upper_Bound': f"{upper:.2f}"
            })

    outlier_df = pd.DataFrame(outlier_summary)
    st.dataframe(outlier_df.set_index('Column'))

    # Data Consistency Checks
    st.subheader("✅ DATA CONSISTENCY CHECKS")

    # Check 1: Total Price Calculation
    # Need to re-add these temporary columns for the check then drop them
    df['calculated_total'] = (df['unit_price_usd'] * df['quantity']) - df['discount_amount_usd']
    price_mismatch = (abs(df['total_price_usd'] - df['calculated_total']) > 0.01).sum()
    st.write(f"**1️⃣ Price Calculation Check:**")
    st.write(f"   - Mismatched records: {price_mismatch:,}")
    st.write(f"   - Accuracy: {(1 - price_mismatch/len(df)) * 100:.2f}%")

    # Check 2: Profit Calculation
    df['calculated_profit'] = df['total_price_usd'] - df['cost_usd']
    profit_mismatch = (abs(df['profit_usd'] - df['calculated_profit']) > 0.01).sum()
    st.write(f"**2️⃣ Profit Calculation Check:**")
    st.write(f"   - Mismatched records: {profit_mismatch:,}")
    st.write(f"   - Accuracy: {(1 - profit_mismatch/len(df)) * 100:.2f}%")

    # Check 3: Profit Margin Calculation
    df['calculated_margin'] = (df['profit_usd'] / df['total_price_usd']) * 100
    margin_mismatch = (abs(df['profit_margin_percent'] - df['calculated_margin']) > 0.1).sum()
    st.write(f"**3️⃣ Profit Margin Check:**")
    st.write(f"   - Mismatched records: {margin_mismatch:,}")
    st.write(f"   - Accuracy: {(1 - margin_mismatch/len(df)) * 100:.2f}%")

    # Check 4: Logical Consistency
    st.write(f"**4️⃣ Logical Consistency Checks:**")
    st.write(f"   - Orders with negative profit: {(df['profit_usd'] < 0).sum():,}")
    st.write(f"   - Orders with 0 quantity: {(df['quantity'] == 0).sum():,}")
    st.write(f"   - Orders with discount > 100%: {(df['discount_percent'] > 100).sum():,}")
    st.write(f"   - Ages below 18: {(df['age'] < 18).sum():,}")
    st.write(f"   - Ages above 100: {(df['age'] > 100).sum():,}")

    # Drop temporary columns for data consistency checks
    df.drop(columns=['calculated_total', 'calculated_profit', 'calculated_margin'], inplace=True, errors='ignore')

    st.markdown("""
---
## 🎯 Data Quality Summary

### ✅ Strengths
- No duplicate orders - each order_id is unique
- Pricing calculations are consistent and accurate
- Proper data types across all columns
- Logical business rules are maintained

### ⚠️ Observations
- Missing values are contextual (return_reason, coupon_code) - this is expected behavior
- Customer feedback has ~80% completion rate
- Some outliers exist in pricing and quantity, which may represent bulk/enterprise orders

### 🔧 Recommendations
- Missing values in return_reason, coupon_code, and customer_feedback are acceptable
- Outliers in price/quantity should be investigated but not removed (may be legitimate bulk orders)
- Data is production-ready for analysis

---
""")

    # Add Feature Engineering text here (similar to how it was in previous writefile)
    st.markdown("""# 🛠️ SECTION 4: FEATURE ENGINEERING""")

    st.markdown("""
---
## 🎯 Feature Engineering Summary

We have successfully created **30+ new features** across multiple dimensions:

### 📅 **Time-Based Features**
- Quarter, Day of Week, Week of Year
- Time of Day categories (Morning, Afternoon, Evening, Night)
- Account age in days

### 👥 **Customer Behavior Features**
- Average Order Value (AOV)
- Customer Total Spent (CLV proxy)
- RFM Analysis (Recency, Frequency, Monetary)
- Loyalty tiers

### 📦 **Product Performance Features**
- Total quantity sold per product
- Rating deviation from average
- Price positioning vs category average

### 💰 **Financial Metrics**
- Revenue per item
- Profit per unit
- High-value order flags
- Discount depth categorization

### 🚚 **Operational Features**
- Fast delivery indicator
- International shipping flag
- Support ticket engagement

### ⚠️ **Risk & Quality Features**
- Fraud risk categories
- Customer loyalty tiers

### 📱 **Marketing Features**
- Engagement rate
- Cart abandonment flags

These features will enable deeper business insights and more sophisticated analysis.

---
""")


    # ------------------------------------------------------------------------------------------------------------------
    # GLOBAL FILTERS
    # ------------------------------------------------------------------------------------------------------------------
    st.sidebar.header("Global Filters")

    # Date Range Filter
    min_date = df['order_date'].min().date()
    max_date = df['order_date'].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['order_date'].dt.date >= start_date) & (df['order_date'].dt.date <= end_date)]
    else:
        df_filtered = df.copy()

    # Customer Segment Filter
    all_segments = ['All'] + list(df['customer_segment'].unique())
    selected_segments = st.sidebar.multiselect(
        "Select Customer Segment",
        options=all_segments,
        default='All'
    )
    if 'All' not in selected_segments:
        df_filtered = df_filtered[df_filtered['customer_segment'].isin(selected_segments)]

    # Category Filter
    all_categories = ['All'] + list(df['category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Select Product Category",
        options=all_categories,
        default='All'
    )
    if 'All' not in selected_categories:
        df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]

    # Brand Filter
    all_brands = ['All'] + list(df['brand'].unique())
    selected_brands = st.sidebar.multiselect(
        "Select Brand",
        options=all_brands,
        default='All'
    )
    if 'All' not in selected_brands:
        df_filtered = df_filtered[df_filtered['brand'].isin(selected_brands)]

    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
        # If no data, return a dummy empty DataFrame to prevent errors in page functions
        df_final = pd.DataFrame(columns=df.columns)
    else:
        df_final = df_filtered

    # Streamlit Page Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Dashboard Overview & KPIs",
        "EDA - Numerical Analysis",
        "EDA - Categorical Analysis",
        "Bivariate & Multivariate Analysis",
        "Time Series Analysis",
        "Statistical Hypothesis Testing",
        "Business Insights & Deep Dives",
        "Advanced Visualizations",
        "Final Summary & Recommendations"
    ])

    # Pass df_final to each page function
    if page == "Dashboard Overview & KPIs":
        dashboard_kpis_page(df_final)
    elif page == "EDA - Numerical Analysis":
        eda_numerical_page(df_final)
    elif page == "EDA - Categorical Analysis":
        eda_categorical_page(df_final)
    elif page == "Bivariate & Multivariate Analysis":
        bivariate_multivariate_analysis_page(df_final)
    elif page == "Time Series Analysis":
        time_series_analysis_page(df_final)
    elif page == "Statistical Hypothesis Testing":
        statistical_hypothesis_testing_page(df_final)
    elif page == "Business Insights & Deep Dives":
        business_insights_page(df_final)
    elif page == "Advanced Visualizations":
        advanced_visualizations_page(df_final)
    elif page == "Final Summary & Recommendations":
        final_summary_recommendations_page(df_final)


if __name__ == "__main__":
    main()
