# Core libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

# Streamlit
import streamlit as st

# Display configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# Plotly template
pio.templates.default = "plotly_white"

# Color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8'
}

# ----------------------------------------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Convert date columns to datetime objects
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['account_creation_date'] = pd.to_datetime(df['account_creation_date'])

    # Feature Engineering (as implemented in the notebook)
    # 1. Temporal features
    df['order_dayofweek'] = df['order_date'].dt.dayofweek
    df['order_weekofyear'] = df['order_date'].dt.isocalendar().week.astype(int)
    df['order_quarter'] = df['order_date'].dt.quarter
    df['day_name'] = df['order_date'].dt.day_name()
    df['month_name'] = df['order_date'].dt.month_name()
    df['is_month_start'] = df['order_date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['order_date'].dt.is_month_end.astype(int)
    df['year_month'] = df['order_date'].dt.to_period('M').astype(str)

    # 2. Customer tenure
    df['customer_tenure_days'] = (df['order_date'] - df['account_creation_date']).dt.days
    df['customer_tenure_months'] = df['customer_tenure_days'] / 30.44

    # 3. Order value categories
    df['order_value_category'] = pd.cut(df['total_price_usd'],
                                          bins=[0, 50, 150, 300, 1000, float('inf')],
                                          labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                          right=False)

    # 4. Discount effectiveness
    df['discount_efficiency'] = df['profit_usd'] / (df['discount_amount_usd'] + 1)
    df['has_discount'] = (df['discount_percent'] > 0).astype(int)

    # 5. Customer value metrics
    # Handle potential division by zero for total_orders_by_customer
    df['average_order_value'] = df['total_price_usd'] / df['total_orders_by_customer'].replace(0, np.nan)
    df['customer_lifetime_value'] = df['total_price_usd'] * df['total_orders_by_customer']

    # 6. Product performance
    df['revenue_per_unit'] = df['total_price_usd'] / df['quantity'].replace(0, np.nan)
    df['profit_per_unit'] = df['profit_usd'] / df['quantity'].replace(0, np.nan)

    # 7. Shipping efficiency
    df['shipping_cost_ratio'] = df['shipping_cost_usd'] / df['total_price_usd'].replace(0, np.nan)
    # Ensure std() is not 0 for delivery_efficiency calculation
    df['delivery_efficiency'] = df.groupby('shipping_method')['delivery_days'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )

    # 8. Risk indicators
    df['is_high_risk'] = (df['fraud_risk_score'] > 70).astype(int)
    df['is_returned'] = (df['order_status'] == 'Returned').astype(int)
    df['payment_issue'] = (df['payment_status'].isin(['Failed', 'Pending'])).astype(int)

    # 9. Marketing attribution
    df['is_paid_channel'] = df['campaign_source'].isin(['Google Ads', 'Facebook']).astype(int)
    df['session_engagement'] = df['pages_visited'] / (df['session_duration_minutes'] + 1)

    # 10. Time-based segments
    df['time_of_day'] = pd.cut(df['order_hour'],
                                bins=[0, 6, 12, 18, 24],
                                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                include_lowest=True)

    # 11. RFM Segmentation
    # Calculate metrics per customer
    customer_metrics = df.groupby('customer_id').agg({
        'order_date': 'max',  # Recency
        'order_id': 'count',  # Frequency
        'total_price_usd': 'sum'  # Monetary
    }).reset_index()

    customer_metrics.columns = ['customer_id', 'last_order_date', 'order_count', 'total_spent']

    # Calculate recency in days from the max date
    max_date = df['order_date'].max()
    customer_metrics['recency_days'] = (max_date - customer_metrics['last_order_date']).dt.days

    # Create quintile scores
    num_quantiles = 5 # Desired number of quantiles

    # Recency: higher score for lower recency (more recent)
    customer_metrics['recency_score'] = pd.qcut(customer_metrics['recency_days'], num_quantiles, labels=False, duplicates='drop')
    customer_metrics['recency_score'] = customer_metrics['recency_score'].max() - customer_metrics['recency_score'] + 1

    # Frequency: higher score for higher frequency
    unique_frequency_bins = pd.qcut(customer_metrics['order_count'], num_quantiles, retbins=True, duplicates='drop')[1]
    n_bins_frequency = len(unique_frequency_bins) - 1
    if n_bins_frequency > 0:
        frequency_labels = list(range(1, n_bins_frequency + 1))
        customer_metrics['frequency_score'] = pd.qcut(customer_metrics['order_count'], num_quantiles, labels=frequency_labels, duplicates='drop')
    else:
        customer_metrics['frequency_score'] = 1

    # Monetary: higher score for higher monetary value
    unique_monetary_bins = pd.qcut(customer_metrics['total_spent'], num_quantiles, retbins=True, duplicates='drop')[1]
    n_bins_monetary = len(unique_monetary_bins) - 1
    if n_bins_monetary > 0:
        monetary_labels = list(range(1, num_quantiles + 1))
        customer_metrics['monetary_score'] = pd.qcut(customer_metrics['total_spent'], num_quantiles, labels=monetary_labels, duplicates='drop')
    else:
        customer_metrics['monetary_score'] = 1

    # Calculate RFM score
    customer_metrics['rfm_score'] = (
        customer_metrics['recency_score'].astype(int) +
        customer_metrics['frequency_score'].astype(int) +
        customer_metrics['monetary_score'].astype(int)
    )

    # Create segments
    def rfm_segment(row):
        if row['rfm_score'] >= 13:
            return 'Champions'
        elif row['rfm_score'] >= 10:
            return 'Loyal Customers'
        elif row['rfm_score'] >= 7:
            return 'Potential Loyalists'
        elif row['rfm_score'] >= 5:
            return 'At Risk'
        else:
            return 'Lost'

    customer_metrics['rfm_segment'] = customer_metrics.apply(rfm_segment, axis=1)

    # Merge RFM segments back to the main DataFrame
    df = df.merge(customer_metrics[['customer_id', 'rfm_segment']], on='customer_id', how='left')

    # Age group for customer analysis
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 25, 35, 45, 55, 100],
                             labels=['18-25', '26-35', '36-45', '46-55', '56+'])


    return df

# ----------------------------------------------------------------------------
# Dashboard Sections as Functions
# ----------------------------------------------------------------------------

def section_revenue_profitability(df_filtered):
    st.header("📈 Revenue & Profitability Analysis")

    # 1. Monthly Revenue Trend
    st.subheader("Monthly Revenue Trend")
    monthly_data = df_filtered.groupby('year_month')['total_price_usd'].sum().reset_index()
    fig_monthly_revenue = go.Figure()
    fig_monthly_revenue.add_trace(go.Scatter(
        x=monthly_data['year_month'], y=monthly_data['total_price_usd'],
        mode='lines+markers', name='Monthly Revenue', line=dict(color=COLORS['primary'], width=2)
    ))
    fig_monthly_revenue.update_layout(title='Monthly Revenue Trend',
                                      xaxis_title='Month',
                                      yaxis_title='Total Revenue (USD)',
                                      hovermode='x unified')
    st.plotly_chart(fig_monthly_revenue, use_container_width=True)

    # 2. Daily Revenue Trend with 7-Day Moving Average
    st.subheader("Daily Revenue Trend with 7-Day Moving Average")
    daily_metrics = df_filtered.groupby(df_filtered['order_date'].dt.date).agg({
        'total_price_usd': 'sum',
    }).reset_index()
    daily_metrics.columns = ['date', 'revenue']
    daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
    daily_metrics['revenue_ma7'] = daily_metrics['revenue'].rolling(window=7).mean()

    fig_daily_revenue = go.Figure()
    fig_daily_revenue.add_trace(go.Scatter(
        x=daily_metrics['date'], y=daily_metrics['revenue'],
        mode='lines', name='Daily Revenue', line=dict(color=COLORS['primary'], width=1), opacity=0.7
    ))
    fig_daily_revenue.add_trace(go.Scatter(
        x=daily_metrics['date'], y=daily_metrics['revenue_ma7'],
        mode='lines', name='7-Day Moving Average', line=dict(color=COLORS['danger'], width=3)
    ))
    fig_daily_revenue.update_layout(title='Daily Revenue Trend with 7-Day Moving Average',
                                     xaxis_title='Date',
                                     yaxis_title='Revenue (USD)',
                                     hovermode='x unified')
    st.plotly_chart(fig_daily_revenue, use_container_width=True)

    # 3. Revenue & Profit by Customer Segment
    st.subheader("Revenue & Profit by Customer Segment")
    segment_analysis = df_filtered.groupby('customer_segment').agg({
        'total_price_usd': 'sum',
        'profit_usd': 'sum'
    }).reset_index()

    fig_segment = make_subplots(rows=1, cols=2, subplot_titles=('Total Revenue by Segment', 'Total Profit by Segment'))
    fig_segment.add_trace(go.Bar(x=segment_analysis['customer_segment'], y=segment_analysis['total_price_usd'],
                                  marker_color=COLORS['primary'], name='Revenue'), row=1, col=1)
    fig_segment.add_trace(go.Bar(x=segment_analysis['customer_segment'], y=segment_analysis['profit_usd'],
                                  marker_color=COLORS['success'], name='Profit'), row=1, col=2)
    fig_segment.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_segment, use_container_width=True)

    # 4. Top N Categories by Revenue & Profit
    st.subheader("Top Categories by Revenue & Profit")
    def get_top_n_data(df_input, group_col, n):
        grouped_data = df_input.groupby(group_col).agg({
            'total_price_usd': 'sum',
            'profit_usd': 'sum'
        }).nlargest(n, 'total_price_usd').reset_index()
        return grouped_data

    if not df_filtered.empty and 'category' in df_filtered.columns:
        max_categories = len(df_filtered['category'].unique())
        top_n_categories = st.slider("Select Top N Categories", min_value=1, max_value=max(1, max_categories), value=min(10, max_categories), key='top_n_cat')
        categories_data = get_top_n_data(df_filtered, 'category', top_n_categories)

        if not categories_data.empty:
            fig_categories = make_subplots(rows=1, cols=2, subplot_titles=(f'Top {top_n_categories} Categories by Revenue', f'Top {top_n_categories} Categories by Profit'))
            fig_categories.add_trace(go.Bar(y=categories_data['category'], x=categories_data['total_price_usd'],
                                            orientation='h', marker_color=COLORS['primary'], name='Revenue'), row=1, col=1)
            fig_categories.add_trace(go.Bar(y=categories_data['category'], x=categories_data['profit_usd'],
                                            orientation='h', marker_color=COLORS['success'], name='Profit'), row=1, col=2)
            fig_categories.update_layout(height=500, showlegend=False)
            fig_categories.update_yaxes(categoryorder='total ascending', row=1, col=1)
            fig_categories.update_yaxes(categoryorder='total ascending', row=1, col=2)
            st.plotly_chart(fig_categories, use_container_width=True)
        else:
            st.info("No category data available based on current filters.")
    else:
        st.info("No category data available based on current filters.")

    # 5. Top N Countries by Revenue & Profit
    st.subheader("Top Countries by Revenue & Profit")
    if not df_filtered.empty and 'country' in df_filtered.columns:
        max_countries = len(df_filtered['country'].unique())
        top_n_countries = st.slider("Select Top N Countries", min_value=1, max_value=max(1, max_countries), value=min(10, max_countries), key='top_n_country')
        countries_data = get_top_n_data(df_filtered, 'country', top_n_countries)

        if not countries_data.empty:
            fig_countries = make_subplots(rows=1, cols=2, subplot_titles=(f'Top {top_n_countries} Countries by Revenue', f'Top {top_n_countries} Countries by Profit'))
            fig_countries.add_trace(go.Bar(y=countries_data['country'], x=countries_data['total_price_usd'],
                                            orientation='h', marker_color=COLORS['info'], name='Revenue'), row=1, col=1)
            fig_countries.add_trace(go.Bar(y=countries_data['country'], x=countries_data['profit_usd'],
                                            orientation='h', marker_color=COLORS['warning'], name='Profit'), row=1, col=2)
            fig_countries.update_layout(height=500, showlegend=False)
            fig_countries.update_yaxes(categoryorder='total ascending', row=1, col=1)
            fig_countries.update_yaxes(categoryorder='total ascending', row=1, col=2)
            st.plotly_chart(fig_countries, use_container_width=True)
        else:
            st.info("No country data available based on current filters.")
    else:
        st.info("No country data available based on current filters.")

    # 6. Revenue by Day of Week & Order Volume by Hour
    st.subheader("Temporal Patterns: Day of Week & Hour")
    if not df_filtered.empty:
        dow_analysis = df_filtered.groupby('day_name').agg({
            'total_price_usd': 'sum'
        }).reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_analysis['day_name'] = pd.Categorical(dow_analysis['day_name'], categories=day_order, ordered=True)
        dow_analysis = dow_analysis.sort_values('day_name')

        hour_analysis = df_filtered.groupby('order_hour').agg({
            'order_id': 'count'
        }).reset_index()

        fig_temporal = make_subplots(rows=1, cols=2, subplot_titles=('Revenue by Day of Week', 'Order Volume by Hour of Day'))
        fig_temporal.add_trace(go.Bar(x=dow_analysis['day_name'], y=dow_analysis['total_price_usd'],
                                      marker_color=COLORS['secondary'], name='Revenue'), row=1, col=1)
        fig_temporal.add_trace(go.Scatter(x=hour_analysis['order_hour'], y=hour_analysis['order_id'],
                                          mode='lines+markers', marker_color=COLORS['danger'], name='Orders'), row=1, col=2)
        fig_temporal.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_temporal, use_container_width=True)
    else:
        st.info("No data available for temporal analysis based on current filters.")

    # 7. Monthly Revenue Comparison Across Years (Seasonality)
    st.subheader("Monthly Revenue Comparison Across Years (Seasonality)")
    if not df_filtered.empty and 'order_year' in df_filtered.columns and 'month_name' in df_filtered.columns:
        month_year_analysis = df_filtered.groupby(['order_year', 'month_name']).agg({
            'total_price_usd': 'sum'
        }).reset_index()

        month_pivot = month_year_analysis.pivot(index='month_name', columns='order_year', values='total_price_usd')
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        month_pivot = month_pivot.reindex(month_order)

        fig_seasonality = go.Figure()
        for year in month_pivot.columns:
            fig_seasonality.add_trace(go.Scatter(
                x=month_pivot.index,
                y=month_pivot[year],
                mode='lines+markers',
                name=f'Year {year}',
                line=dict(width=3),
                marker=dict(size=8)
            ))
        fig_seasonality.update_layout(title='Monthly Revenue Comparison Across Years (Seasonality)',
                                       xaxis_title='Month',
                                       yaxis_title='Total Revenue (USD)',
                                       hovermode='x unified')
        st.plotly_chart(fig_seasonality, use_container_width=True)
    else:
        st.info("No data available for seasonality analysis based on current filters.")

def section_customer_behavior(df_filtered):
    st.header("👥 Customer Behavior & Segmentation")

    # 1. Customer Distribution by RFM Segment
    st.subheader("Customer Distribution by RFM Segment")
    if not df_filtered.empty and 'rfm_segment' in df_filtered.columns:
        rfm_counts = df_filtered['rfm_segment'].value_counts().reset_index()
        rfm_counts.columns = ['rfm_segment', 'count']
        fig_rfm_segment = go.Figure(data=[go.Pie(
            labels=rfm_counts['rfm_segment'],
            values=rfm_counts['count'],
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        fig_rfm_segment.update_layout(title='Customer Distribution by RFM Segment')
        st.plotly_chart(fig_rfm_segment, use_container_width=True)
    else:
        st.info("No RFM segment data available based on current filters.")

    # 2. Avg Customer Lifetime Value by Customer Segment
    st.subheader("Average Customer Lifetime Value by Customer Segment")
    if not df_filtered.empty and 'customer_segment' in df_filtered.columns and 'customer_lifetime_value' in df_filtered.columns:
        clv_segment = df_filtered.groupby('customer_segment')['customer_lifetime_value'].mean().reset_index()
        fig_clv_segment = go.Figure(data=[go.Bar(
            x=clv_segment['customer_segment'], y=clv_segment['customer_lifetime_value'],
            marker_color=COLORS['primary']
        )])
        fig_clv_segment.update_layout(title='Average Customer Lifetime Value by Customer Segment',
                                        xaxis_title='Customer Segment',
                                        yaxis_title='Average CLV (USD)')
        st.plotly_chart(fig_clv_segment, use_container_width=True)
    else:
        st.info("No customer lifetime value data available based on current filters.")

    # 3. Customer Loyalty Distribution
    st.subheader("Customer Loyalty Score Distribution")
    if not df_filtered.empty and 'customer_loyalty_score' in df_filtered.columns:
        fig_loyalty_dist = go.Figure(data=[go.Histogram(
            x=df_filtered['customer_loyalty_score'],
            nbinsx=30,
            marker_color=COLORS['secondary']
        )])
        fig_loyalty_dist.update_layout(title='Customer Loyalty Score Distribution',
                                         xaxis_title='Loyalty Score',
                                         yaxis_title='Count')
        st.plotly_chart(fig_loyalty_dist, use_container_width=True)
    else:
        st.info("No customer loyalty score data available based on current filters.")

    # 4. Age Distribution
    st.subheader("Age Distribution")
    if not df_filtered.empty and 'age' in df_filtered.columns:
        fig_age_dist = go.Figure(data=[go.Histogram(
            x=df_filtered['age'],
            nbinsx=30,
            marker_color=COLORS['success']
        )])
        fig_age_dist.update_layout(title='Age Distribution',
                                     xaxis_title='Age',
                                     yaxis_title='Count')
        st.plotly_chart(fig_age_dist, use_container_width=True)
    else:
        st.info("No age data available based on current filters.")

    # 5. Gender Distribution
    st.subheader("Gender Distribution")
    if not df_filtered.empty and 'gender' in df_filtered.columns:
        gender_counts = df_filtered['gender'].value_counts()
        fig_gender_dist = go.Figure(data=[go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        fig_gender_dist.update_layout(title='Gender Distribution')
        st.plotly_chart(fig_gender_dist, use_container_width=True)
    else:
        st.info("No gender data available based on current filters.")

    # 6. Avg Order Value by Customer Tenure (binned)
    st.subheader("Average Order Value by Customer Tenure")
    if not df_filtered.empty and 'customer_tenure_months' in df_filtered.columns:
        tenure_bins = pd.cut(df_filtered['customer_tenure_months'],
                             bins=[0, 6, 12, 24, 100],
                             labels=['0-6 months', '7-12 months', '13-24 months', '24+ months'])
        tenure_aov = df_filtered.groupby(tenure_bins)['total_price_usd'].mean().reset_index()
        fig_tenure_aov = go.Figure(data=[go.Bar(
            x=tenure_aov['customer_tenure_months'], y=tenure_aov['total_price_usd'],
            marker_color=COLORS['info']
        )])
        fig_tenure_aov.update_layout(title='Average Order Value by Customer Tenure',
                                       xaxis_title='Customer Tenure',
                                       yaxis_title='Average Order Value (USD)')
        st.plotly_chart(fig_tenure_aov, use_container_width=True)
    else:
        st.info("No customer tenure data available based on current filters.")

    # 7. Loyalty Score by Age Group
    st.subheader("Average Loyalty Score by Age Group")
    if not df_filtered.empty and 'age_group' in df_filtered.columns and 'customer_loyalty_score' in df_filtered.columns:
        age_loyalty = df_filtered.groupby('age_group')['customer_loyalty_score'].mean().reset_index()
        fig_age_loyalty = go.Figure(data=[go.Bar(
            x=age_loyalty['age_group'], y=age_loyalty['customer_loyalty_score'],
            marker_color=COLORS['warning']
        )])
        fig_age_loyalty.update_layout(title='Average Loyalty Score by Age Group',
                                       xaxis_title='Age Group',
                                       yaxis_title='Average Loyalty Score')
        st.plotly_chart(fig_age_loyalty, use_container_width=True)
    else:
        st.info("No age group or loyalty score data available based on current filters.")

def section_product_performance(df_filtered):
    st.header("📦 Product Performance Deep Dive")

    # 1. Top N Products by Revenue and Profit
    st.subheader("Top Products by Revenue and Profit")
    if not df_filtered.empty and 'product_id' in df_filtered.columns and 'product_name' in df_filtered.columns:
        product_performance = df_filtered.groupby(['product_id', 'product_name', 'category', 'brand']).agg(
            Total_Revenue=('total_price_usd', 'sum'),
            Total_Profit=('profit_usd', 'sum'),
            Units_Sold=('quantity', 'sum'),
            Order_Count=('order_id', 'count'),
            Avg_Rating=('product_rating_avg', 'mean'),
            Returned_Count=('is_returned', 'sum')
        ).reset_index()
        product_performance['Return_Rate'] = (product_performance['Returned_Count'] / product_performance['Order_Count']) * 100

        if not product_performance.empty:
            max_products = len(product_performance)
            top_n_products = st.slider("Select Top N Products", min_value=1, max_value=max(1, max_products), value=min(15, max_products), key='top_n_products')
            top_products = product_performance.nlargest(top_n_products, 'Total_Revenue').round(2)
            st.write(top_products[['product_name', 'category', 'brand', 'Total_Revenue', 'Total_Profit', 'Units_Sold', 'Avg_Rating', 'Return_Rate']])

            fig_top_products = make_subplots(rows=1, cols=2,
                                             subplot_titles=[f'Top {top_n_products} Products by Revenue',
                                                             f'Top {top_n_products} Products by Profit'])

            fig_top_products.add_trace(go.Bar(y=top_products['product_name'], x=top_products['Total_Revenue'],
                                              orientation='h', marker_color=COLORS['primary'], name='Revenue'), row=1, col=1)
            fig_top_products.add_trace(go.Bar(y=top_products['product_name'], x=top_products['Total_Profit'],
                                              orientation='h', marker_color=COLORS['success'], name='Profit'), row=1, col=2)

            fig_top_products.update_layout(height=600, showlegend=False)
            fig_top_products.update_yaxes(categoryorder='total ascending', row=1, col=1)
            fig_top_products.update_yaxes(categoryorder='total ascending', row=1, col=2)
            st.plotly_chart(fig_top_products, use_container_width=True)
        else:
            st.info("No product data available based on current filters.")
    else:
        st.info("No product data available based on current filters.")


    # 2. Top N Brands by Revenue and Profit
    st.subheader("Top Brands by Revenue and Profit")
    if not df_filtered.empty and 'brand' in df_filtered.columns:
        brand_performance = df_filtered.groupby('brand').agg(
            Total_Revenue=('total_price_usd', 'sum'),
            Total_Profit=('profit_usd', 'sum'),
            Avg_Profit_Margin=('profit_margin_percent', 'mean'),
            Order_Count=('order_id', 'count'),
            Avg_Rating=('product_rating_avg', 'mean')
        ).reset_index()

        if not brand_performance.empty:
            max_brands = len(brand_performance)
            top_n_brands = st.slider("Select Top N Brands", min_value=1, max_value=max(1, max_brands), value=min(15, max_brands), key='top_n_brands')
            top_brands = brand_performance.nlargest(top_n_brands, 'Total_Revenue').round(2)
            st.write(top_brands[['brand', 'Total_Revenue', 'Total_Profit', 'Avg_Profit_Margin', 'Avg_Rating']])

            fig_top_brands = make_subplots(rows=1, cols=2,
                                           subplot_titles=[f'Top {top_n_brands} Brands by Revenue',
                                                           f'Top {top_n_brands} Brands by Profit'])

            fig_top_brands.add_trace(go.Bar(y=top_brands['brand'], x=top_brands['Total_Revenue'],
                                             orientation='h', marker_color=COLORS['secondary'], name='Revenue'), row=1, col=1)
            fig_top_brands.add_trace(go.Bar(y=top_brands['brand'], x=top_brands['Total_Profit'],
                                             orientation='h', marker_color=COLORS['warning'], name='Profit'), row=1, col=2)

            fig_top_brands.update_layout(height=600, showlegend=False)
            fig_top_brands.update_yaxes(categoryorder='total ascending', row=1, col=1)
            fig_top_brands.update_yaxes(categoryorder='total ascending', row=1, col=2)
            st.plotly_chart(fig_top_brands, use_container_width=True)
        else:
            st.info("No brand data available based on current filters.")
    else:
        st.info("No brand data available based on current filters.")


    # 3. Top N Sub-Categories by Revenue and Profit
    st.subheader("Top Sub-Categories by Revenue and Profit")
    if not df_filtered.empty and 'sub_category' in df_filtered.columns:
        subcategory_performance = df_filtered.groupby('sub_category').agg(
            Total_Revenue=('total_price_usd', 'sum'),
            Total_Profit=('profit_usd', 'sum'),
            Avg_Profit_Margin=('profit_margin_percent', 'mean'),
            Order_Count=('order_id', 'count'),
            Returned_Count=('is_returned', 'sum')
        ).reset_index()
        subcategory_performance['Return_Rate'] = (subcategory_performance['Returned_Count'] / subcategory_performance['Order_Count']) * 100

        if not subcategory_performance.empty:
            max_subcategories = len(subcategory_performance)
            top_n_subcategories = st.slider("Select Top N Sub-Categories", min_value=1, max_value=max(1, max_subcategories), value=min(15, max_subcategories), key='top_n_subcategories')
            top_subcategories = subcategory_performance.nlargest(top_n_subcategories, 'Total_Revenue').round(2)
            st.write(top_subcategories[['sub_category', 'Total_Revenue', 'Total_Profit', 'Avg_Profit_Margin', 'Return_Rate']])

            fig_top_subcategories = make_subplots(rows=1, cols=2,
                                                subplot_titles=[f'Top {top_n_subcategories} Sub-Categories by Revenue',
                                                                f'Top {top_n_subcategories} Sub-Categories by Profit'])

            fig_top_subcategories.add_trace(go.Bar(y=top_subcategories['sub_category'], x=top_subcategories['Total_Revenue'],
                                                   orientation='h', marker_color=COLORS['info'], name='Revenue'), row=1, col=1)
            fig_top_subcategories.add_trace(go.Bar(y=top_subcategories['sub_category'], x=top_subcategories['Total_Profit'],
                                                   orientation='h', marker_color=COLORS['danger'], name='Profit'), row=1, col=2)

            fig_top_subcategories.update_layout(height=600, showlegend=False)
            fig_top_subcategories.update_yaxes(categoryorder='total ascending', row=1, col=1)
            fig_top_subcategories.update_yaxes(categoryorder='total ascending', row=1, col=2)
            st.plotly_chart(fig_top_subcategories, use_container_width=True)
        else:
            st.info("No sub-category data available based on current filters.")
    else:
        st.info("No sub-category data available based on current filters.")


    # 4. Product Rating Impact on Sales and Returns
    st.subheader("Product Rating Impact on Sales and Returns")
    if not df_filtered.empty and 'product_rating_avg' in df_filtered.columns:
        # Ensure product_rating_avg is not empty before binning
        if not df_filtered['product_rating_avg'].dropna().empty:
            rating_bins = pd.cut(df_filtered['product_rating_avg'],
                                 bins=[0, 2, 3, 4, 5],
                                 labels=['Poor (0-2)', 'Fair (2-3)', 'Good (3-4)', 'Excellent (4-5)'],
                                 right=False) # Use right=False to match notebook's binning style

            rating_impact = df_filtered.groupby(rating_bins).agg(
                Total_Revenue=('total_price_usd', 'sum'),
                Avg_Order_Value=('total_price_usd', 'mean'),
                Order_Count=('order_id', 'count'),
                Returned_Count=('is_returned', 'sum')
            ).reset_index()
            rating_impact['Return_Rate'] = (rating_impact['Returned_Count'] / rating_impact['Order_Count']) * 100
            rating_impact.columns = ['Rating_Group', 'Total_Revenue', 'Avg_Order_Value', 'Order_Count', 'Returned_Count', 'Return_Rate']

            if not rating_impact.empty:
                st.write(rating_impact[['Rating_Group', 'Total_Revenue', 'Avg_Order_Value', 'Order_Count', 'Return_Rate']].round(2))

                fig_rating_impact = make_subplots(rows=1, cols=2,
                                                subplot_titles=['Total Revenue by Product Rating', 'Return Rate by Product Rating'])

                fig_rating_impact.add_trace(go.Bar(x=rating_impact['Rating_Group'], y=rating_impact['Total_Revenue'],
                                                    marker_color=COLORS['primary'], name='Total Revenue'), row=1, col=1)
                fig_rating_impact.add_trace(go.Bar(x=rating_impact['Rating_Group'], y=rating_impact['Return_Rate'],
                                                    marker_color=COLORS['danger'], name='Return Rate (%)'), row=1, col=2)

                fig_rating_impact.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_rating_impact, use_container_width=True)
            else:
                st.info("No product rating data available to analyze impact based on current filters.")
        else:
            st.info("No product rating data available to analyze impact based on current filters.")
    else:
        st.info("No product rating data available based on current filters.")

def section_pricing_discount(df_filtered):
    st.header("🏷️ Pricing & Discount Strategy")

    # 1. Discount Impact on Order Value and Profitability
    st.subheader("Discount Impact on Order Value and Profitability")
    if not df_filtered.empty and 'has_discount' in df_filtered.columns and 'total_price_usd' in df_filtered.columns and 'profit_margin_percent' in df_filtered.columns:
        comparison_data = df_filtered.groupby('has_discount').agg({
            'total_price_usd': 'mean',
            'profit_margin_percent': 'mean'
        }).reset_index()
        comparison_data['discount_label'] = comparison_data['has_discount'].map({0: 'No Discount', 1: 'With Discount'})

        fig_discount_impact = make_subplots(rows=1, cols=2,
            subplot_titles=['Avg Order Value: Discount vs No Discount',
                            'Avg Profit Margin: Discount vs No Discount']
        )

        fig_discount_impact.add_trace(
            go.Bar(x=comparison_data['discount_label'], y=comparison_data['total_price_usd'],
                   marker_color=[COLORS['primary'], COLORS['secondary']],
                   text=comparison_data['total_price_usd'].round(2), textposition='auto'),
            row=1, col=1
        )

        fig_discount_impact.add_trace(
            go.Bar(x=comparison_data['discount_label'], y=comparison_data['profit_margin_percent'],
                   marker_color=[COLORS['success'], COLORS['warning']],
                   text=comparison_data['profit_margin_percent'].round(2), textposition='auto'),
            row=1, col=2
        )

        fig_discount_impact.update_layout(
            title_text="Discount Impact on Order Value and Profitability",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_discount_impact, use_container_width=True)
    else:
        st.info("No discount data available based on current filters.")

    # 2. Performance by Discount Level
    st.subheader("Performance by Discount Level (for discounted orders)")
    if not df_filtered.empty and 'discount_percent' in df_filtered.columns:
        discounted_df = df_filtered[df_filtered['discount_percent'] > 0].copy()

        if not discounted_df.empty:
            discount_levels = pd.cut(discounted_df['discount_percent'],
                                     bins=[0, 5, 10, 15, 20, 100],
                                     labels=['1-5%', '6-10%', '11-15%', '16-20%', '20%+'],
                                     right=False)

            discount_level_analysis = discounted_df.groupby(discount_levels).agg({
                'total_price_usd': 'mean',
                'profit_margin_percent': 'mean',
                'order_id': 'count'
            }).round(2).reset_index()
            discount_level_analysis.columns = ['Discount_Level', 'Avg_Order_Value', 'Avg_Profit_Margin', 'Order_Count']

            fig_discount_levels = make_subplots(rows=1, cols=2,
                subplot_titles=['Avg Order Value by Discount Level', 'Avg Profit Margin by Discount Level']
            )

            fig_discount_levels.add_trace(
                go.Bar(x=discount_level_analysis['Discount_Level'], y=discount_level_analysis['Avg_Order_Value'],
                       marker_color=COLORS['primary'],
                       text=discount_level_analysis['Avg_Order_Value'].round(2), textposition='auto'),
                row=1, col=1
            )

            fig_discount_levels.add_trace(
                go.Bar(x=discount_level_analysis['Discount_Level'], y=discount_level_analysis['Avg_Profit_Margin'],
                       marker_color=COLORS['success'],
                       text=discount_level_analysis['Avg_Profit_Margin'].round(2), textposition='auto'),
                row=1, col=2
            )

            fig_discount_levels.update_layout(
                title_text="Performance Metrics by Discount Level",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_discount_levels, use_container_width=True)
        else:
            st.info("No discounted orders to display based on current filters.")
    else:
        st.info("No discount data available based on current filters.")

def section_operational_metrics(df_filtered):
    st.header("📦 Operational Metrics (Shipping, Returns, Fraud)")

    # 1. Order Status Distribution
    st.subheader("Order Status Distribution")
    if not df_filtered.empty and 'order_status' in df_filtered.columns:
        status_counts = df_filtered['order_status'].value_counts()
        fig_status_dist = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        fig_status_dist.update_layout(title='Order Status Distribution')
        st.plotly_chart(fig_status_dist, use_container_width=True)
    else:
        st.info("No order status data available based on current filters.")

    # 2. Return Reasons Breakdown
    st.subheader("Return Reasons Breakdown")
    if not df_filtered.empty and 'order_status' in df_filtered.columns and 'return_reason' in df_filtered.columns:
        returned_orders_df = df_filtered[df_filtered['order_status'] == 'Returned'].copy()
        if not returned_orders_df.empty:
            return_reasons = returned_orders_df['return_reason'].value_counts()
            fig_return_reasons = go.Figure(data=[go.Pie(
                labels=return_reasons.index,
                values=return_reasons.values,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            fig_return_reasons.update_layout(title='Distribution of Return Reasons')
            st.plotly_chart(fig_return_reasons, use_container_width=True)
        else:
            st.info("No returned orders to analyze return reasons based on current filters.")
    else:
        st.info("No return data available based on current filters.")

    # 3. Return Rate by Category
    st.subheader("Return Rate by Category")
    if not df_filtered.empty and 'category' in df_filtered.columns and 'is_returned' in df_filtered.columns:
        category_returns = df_filtered.groupby('category').agg({
            'order_id': 'count',
            'is_returned': 'sum'
        }).reset_index()
        category_returns['return_rate'] = (category_returns['is_returned'] / category_returns['order_id']) * 100
        category_returns = category_returns.sort_values('return_rate', ascending=False)

        if not category_returns.empty:
            fig_cat_returns = go.Figure(data=[go.Bar(
                y=category_returns['category'],
                x=category_returns['return_rate'],
                orientation='h',
                marker_color=COLORS['danger'],
                text=category_returns['return_rate'].round(2),
                textposition='auto'
            )])
            fig_cat_returns.update_layout(title='Return Rate by Category',
                                          xaxis_title='Return Rate (%)',
                                          yaxis_title='Category',
                                          yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_cat_returns, use_container_width=True)
        else:
            st.info("No categories with return data based on current filters.")
    else:
        st.info("No return data available based on current filters.")

    # 4. Delivery Status Distribution
    st.subheader("Delivery Status Distribution")
    if not df_filtered.empty and 'delivery_status' in df_filtered.columns:
        delivery_status_counts = df_filtered['delivery_status'].value_counts()
        fig_delivery_status = go.Figure(data=[go.Pie(
            labels=delivery_status_counts.index,
            values=delivery_status_counts.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        fig_delivery_status.update_layout(title='Delivery Status Distribution')
        st.plotly_chart(fig_delivery_status, use_container_width=True)
    else:
        st.info("No delivery status data available based on current filters.")

    # 5. Shipping Method Performance
    st.subheader("Shipping Method Performance")
    if not df_filtered.empty and 'shipping_method' in df_filtered.columns and 'shipping_cost_usd' in df_filtered.columns and 'delivery_days' in df_filtered.columns:
        shipping_analysis = df_filtered.groupby('shipping_method').agg({
            'shipping_cost_usd': 'mean',
            'delivery_days': 'mean',
            'order_id': 'count',
            'delivery_status': lambda x: (x == 'Delivered').sum() / len(x) * 100
        }).round(2).reset_index()
        shipping_analysis.columns = ['Shipping_Method', 'Avg_Shipping_Cost', 'Avg_Delivery_Days',
                                      'Order_Count', 'Delivery_Success_Rate']

        if not shipping_analysis.empty:
            fig_shipping_perf = make_subplots(rows=1, cols=3,
                subplot_titles=['Avg Delivery Days', 'Avg Shipping Cost', 'Delivery Success Rate']
            )

            fig_shipping_perf.add_trace(
                go.Bar(x=shipping_analysis['Shipping_Method'], y=shipping_analysis['Avg_Delivery_Days'],
                       marker_color=COLORS['primary'], name='Avg Delivery Days'),
                row=1, col=1
            )
            fig_shipping_perf.add_trace(
                go.Bar(x=shipping_analysis['Shipping_Method'], y=shipping_analysis['Avg_Shipping_Cost'],
                       marker_color=COLORS['secondary'], name='Avg Shipping Cost'),
                row=1, col=2
            )
            fig_shipping_perf.add_trace(
                go.Bar(x=shipping_analysis['Shipping_Method'], y=shipping_analysis['Delivery_Success_Rate'],
                       marker_color=COLORS['success'], name='Delivery Success Rate'),
                row=1, col=3
            )

            fig_shipping_perf.update_layout(title_text="Shipping Method Performance", showlegend=False, height=400)
            st.plotly_chart(fig_shipping_perf, use_container_width=True)
        else:
            st.info("No shipping method data available based on current filters.")
    else:
        st.info("No shipping method data available based on current filters.")


    # 6. Warehouse Performance
    st.subheader("Warehouse Performance")
    if not df_filtered.empty and 'warehouse_location' in df_filtered.columns:
        warehouse_analysis = df_filtered.groupby('warehouse_location').agg({
            'delivery_days': 'mean',
            'shipping_cost_usd': 'mean',
            'order_id': 'count',
            'delivery_status': lambda x: (x == 'Failed').sum() / len(x) * 100
        }).round(2).reset_index()

        warehouse_analysis.columns = ['Warehouse', 'Avg_Delivery_Days', 'Avg_Shipping_Cost',
                                       'Orders_Fulfilled', 'Failure_Rate']

        if not warehouse_analysis.empty:
            fig_warehouse_perf = make_subplots(rows=1, cols=2,
                subplot_titles=['Avg Delivery Days by Warehouse', 'Delivery Failure Rate by Warehouse']
            )

            fig_warehouse_perf.add_trace(
                go.Bar(x=warehouse_analysis['Warehouse'], y=warehouse_analysis['Avg_Delivery_Days'],
                       marker_color=COLORS['info'], name='Avg Delivery Days'),
                row=1, col=1
            )
            fig_warehouse_perf.add_trace(
                go.Bar(x=warehouse_analysis['Warehouse'], y=warehouse_analysis['Failure_Rate'],
                       marker_color=COLORS['danger'], name='Failure Rate'),
                row=1, col=2
            )

            fig_warehouse_perf.update_layout(title_text="Warehouse Performance", showlegend=False, height=400)
            st.plotly_chart(fig_warehouse_perf, use_container_width=True)
        else:
            st.info("No warehouse data available based on current filters.")
    else:
        st.info("No warehouse data available based on current filters.")

    # 7. Fraud Risk Distribution
    st.subheader("Fraud Risk Distribution")
    if not df_filtered.empty and 'fraud_risk_score' in df_filtered.columns:
        fig_fraud_dist = go.Figure(data=[go.Histogram(
            x=df_filtered['fraud_risk_score'],
            nbinsx=30,
            marker_color=COLORS['primary']
        )])
        fig_fraud_dist.update_layout(title='Fraud Risk Score Distribution',
                                      xaxis_title='Fraud Risk Score',
                                      yaxis_title='Count')
        st.plotly_chart(fig_fraud_dist, use_container_width=True)
    else:
        st.info("No fraud risk score data available based on current filters.")

    # 8. Payment Issues by Fraud Risk Level
    st.subheader("Payment Issues by Fraud Risk Level")
    if not df_filtered.empty and 'fraud_risk_score' in df_filtered.columns and 'payment_issue' in df_filtered.columns and 'order_id' in df_filtered.columns:
        fraud_bins = pd.cut(df_filtered['fraud_risk_score'],
                            bins=[0, 30, 60, 80, 100],
                            labels=['Low (0-30)', 'Medium (31-60)', 'High (61-80)', 'Critical (81-100)'],
                            right=True)

        fraud_analysis = df_filtered.groupby(fraud_bins).agg({
            'order_id': 'count',
            'payment_issue': 'sum',
        }).reset_index()

        fraud_analysis['payment_issue_rate'] = (fraud_analysis['payment_issue'] / fraud_analysis['order_id']) * 100

        if not fraud_analysis.empty:
            fig_payment_issues = go.Figure(data=[go.Bar(
                x=fraud_analysis['fraud_risk_score'], y=fraud_analysis['payment_issue_rate'],
                marker_color=COLORS['danger'],
                text=fraud_analysis['payment_issue_rate'].round(2),
                textposition='auto'
            )])
            fig_payment_issues.update_layout(title='Payment Issue Rate by Fraud Risk Level',
                                             xaxis_title='Fraud Risk Level',
                                             yaxis_title='Payment Issue Rate (%)')
            st.plotly_chart(fig_payment_issues, use_container_width=True)
        else:
            st.info("No fraud risk data available based on current filters.")
    else:
        st.info("No fraud risk data available based on current filters.")

def section_marketing_performance(df_filtered):
    st.header("📊 Marketing Channel Performance")

    # 1. Revenue and Profit by Campaign Source
    st.subheader("Revenue and Profit by Campaign Source")
    if not df_filtered.empty and 'campaign_source' in df_filtered.columns:
        campaign_analysis = df_filtered.groupby('campaign_source').agg({
            'total_price_usd': 'sum',
            'profit_usd': 'sum'
        }).reset_index()

        if not campaign_analysis.empty:
            fig_campaign = make_subplots(rows=1, cols=2, subplot_titles=('Revenue by Campaign Source', 'Profit by Campaign Source'))
            fig_campaign.add_trace(go.Bar(x=campaign_analysis['campaign_source'], y=campaign_analysis['total_price_usd'],
                                          marker_color=COLORS['primary'], name='Revenue'), row=1, col=1)
            fig_campaign.add_trace(go.Bar(x=campaign_analysis['campaign_source'], y=campaign_analysis['profit_usd'],
                                          marker_color=COLORS['success'], name='Profit'), row=1, col=2)
            fig_campaign.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_campaign, use_container_width=True)
        else:
            st.info("No campaign data available based on current filters.")
    else:
        st.info("No campaign data available based on current filters.")

    # 2. Revenue and Order Count by Traffic Source
    st.subheader("Revenue and Order Count by Traffic Source")
    if not df_filtered.empty and 'traffic_source' in df_filtered.columns:
        traffic_analysis = df_filtered.groupby('traffic_source').agg({
            'total_price_usd': 'sum',
            'order_id': 'count'
        }).reset_index()

        if not traffic_analysis.empty:
            fig_traffic = make_subplots(rows=1, cols=2, subplot_titles=('Revenue by Traffic Source', 'Order Count by Traffic Source'))
            fig_traffic.add_trace(go.Bar(x=traffic_analysis['traffic_source'], y=traffic_analysis['total_price_usd'],
                                         marker_color=COLORS['secondary'], name='Revenue'), row=1, col=1)
            fig_traffic.add_trace(go.Bar(x=traffic_analysis['traffic_source'], y=traffic_analysis['order_id'],
                                         marker_color=COLORS['info'], name='Order Count'), row=1, col=2)
            fig_traffic.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_traffic, use_container_width=True)
        else:
            st.info("No traffic source data available based on current filters.")
    else:
        st.info("No traffic source data available based on current filters.")

    # 3. Revenue and Cart Abandonment Rate by Device Type
    st.subheader("Revenue and Cart Abandonment Rate by Device Type")
    if not df_filtered.empty and 'device_type' in df_filtered.columns and 'abandoned_cart_before' in df_filtered.columns:
        device_analysis = df_filtered.groupby('device_type').agg({
            'total_price_usd': 'sum',
            'abandoned_cart_before': lambda x: (x == 'Yes').sum() / len(x) * 100
        }).round(2).reset_index()
        device_analysis.columns = ['device_type', 'Total_Revenue', 'Abandoned_Cart_Rate']

        if not device_analysis.empty:
            fig_device = make_subplots(rows=1, cols=2, subplot_titles=('Revenue by Device Type', 'Cart Abandonment Rate by Device Type'))
            fig_device.add_trace(go.Bar(x=device_analysis['device_type'], y=device_analysis['Total_Revenue'],
                                        marker_color=COLORS['warning'], name='Revenue'), row=1, col=1)
            fig_device.add_trace(go.Bar(x=device_analysis['device_type'], y=device_analysis['Abandoned_Cart_Rate'],
                                        marker_color=COLORS['danger'], name='Abandoned Cart Rate'), row=1, col=2)
            fig_device.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_device, use_container_width=True)
        else:
            st.info("No device type data available based on current filters.")
    else:
        st.info("No device type data available based on current filters.")

def section_statistical_tests(df_full):
    st.header("🔬 Statistical Hypothesis Testing Results")
    st.markdown("""
This section summarizes the findings from key statistical hypothesis tests performed on the **overall (unfiltered)** dataset. These tests help validate observed patterns and determine their statistical significance.
""")

    # Test 1: Customer Segment Profitability Test
    st.subheader("1. Customer Segment Profitability Test")
    st.markdown("**Hypotheses:**")
    st.markdown("  - **H0:** Mean profit margins are equal across all customer segments")
    st.markdown("  - **H1:** At least one segment has different mean profit margin")

    if not df_full.empty and 'customer_segment' in df_full.columns and 'profit_margin_percent' in df_full.columns:
        regular_pm = df_full[df_full['customer_segment'] == 'Regular']['profit_margin_percent'].dropna()
        premium_pm = df_full[df_full['customer_segment'] == 'Premium']['profit_margin_percent'].dropna()
        vip_pm = df_full[df_full['customer_segment'] == 'VIP']['profit_margin_percent'].dropna()

        if not regular_pm.empty and not premium_pm.empty and not vip_pm.empty:
            f_stat, p_value_segment = f_oneway(regular_pm, premium_pm, vip_pm)

            st.write(f"**F-statistic:** {f_stat:.4f}")
            st.write(f"**P-value:** {p_value_segment:.6f}")

            if p_value_segment < 0.05:
                st.markdown("**Result:** ✓ REJECT null hypothesis (p < 0.05)")
                st.markdown("**Interpretation:** There ARE statistically significant differences in profit margins across customer segments.")
            else:
                st.markdown("**Result:** ✗ FAIL TO REJECT null hypothesis (p >= 0.05)")
                st.markdown("**Interpretation:** There are no statistically significant differences in mean profit margins across customer segments.")

            st.markdown("**Segment Profit Margin Means (Overall Data):**")
            st.write(f"  - Regular: {regular_pm.mean():.2f}%")
            st.write(f"  - Premium: {premium_pm.mean():.2f}%")
            st.write(f"  - VIP: {vip_pm.mean():.2f}%")
        else:
            st.info("Insufficient data to perform ANOVA for customer segment profitability.")
    else:
        st.info("Required columns for customer segment profitability test are missing.")

    # Test 2: Discount Effectiveness Test
    st.subheader("2. Discount Effectiveness Test")
    st.markdown("**Hypotheses:**")
    st.markdown("  - **H0:** Mean profit margin is the same for discounted and non-discounted orders")
    st.markdown("  - **H1:** Mean profit margin differs between discounted and non-discounted orders")

    if not df_full.empty and 'has_discount' in df_full.columns and 'profit_margin_percent' in df_full.columns:
        discount_pm = df_full[df_full['has_discount'] == 1]['profit_margin_percent'].dropna()
        no_discount_pm = df_full[df_full['has_discount'] == 0]['profit_margin_percent'].dropna()

        if not discount_pm.empty and not no_discount_pm.empty:
            t_stat_discount, p_value_discount = ttest_ind(discount_pm, no_discount_pm)

            st.write(f"**T-statistic:** {t_stat_discount:.4f}")
            st.write(f"**P-value:** {p_value_discount:.6f}")

            if p_value_discount < 0.05:
                st.markdown("**Result:** ✓ REJECT null hypothesis (p < 0.05)")
                st.markdown("**Interpretation:** Discounts have a statistically significant impact on profit margins. Specifically, discounted orders tend to have lower profit margins.")
            else:
                st.markdown("**Result:** ✗ FAIL TO REJECT null hypothesis (p >= 0.05)")
                st.markdown("**Interpretation:** There is no statistically significant difference in mean profit margins between discounted and non-discounted orders.")

            st.markdown("**Profit Margin Means (Overall Data):**")
            st.write(f"  - With Discount: {discount_pm.mean():.2f}%")
            st.write(f"  - Without Discount: {no_discount_pm.mean():.2f}%")
            st.write(f"  - Difference: {discount_pm.mean() - no_discount_pm.mean():.2f}%")
        else:
            st.info("Insufficient data to perform T-test for discount effectiveness.")
    else:
        st.info("Required columns for discount effectiveness test are missing.")

    # Test 3: Gender Purchase Behavior Test
    st.subheader("3. Gender Differences in Order Value")
    st.markdown("**Hypotheses:**")
    st.markdown("  - **H0:** Mean order value is the same for males and females")
    st.markdown("  - **H1:** Mean order value differs between males and females")

    if not df_full.empty and 'gender' in df_full.columns and 'total_price_usd' in df_full.columns:
        male_orders = df_full[df_full['gender'] == 'Male']['total_price_usd'].dropna()
        female_orders = df_full[df_full['gender'] == 'Female']['total_price_usd'].dropna()

        if not male_orders.empty and not female_orders.empty:
            t_stat_gender, p_value_gender = ttest_ind(male_orders, female_orders)

            st.write(f"**T-statistic:** {t_stat_gender:.4f}")
            st.write(f"**P-value:** {p_value_gender:.6f}")

            if p_value_gender < 0.05:
                st.markdown("**Result:** ✓ REJECT null hypothesis (p < 0.05)")
                st.markdown("**Interpretation:** There IS a statistically significant difference in average order values between males and females.")
            else:
                st.markdown("**Result:** ✗ FAIL TO REJECT null hypothesis (p >= 0.05)")
                st.markdown("**Interpretation:** There is no statistically significant difference in average order values between males and females.")

            st.markdown("**Order Value Statistics (Overall Data):**")
            st.write(f"  - Male - Mean: ${male_orders.mean():.2f}")
            st.write(f"  - Female - Mean: ${female_orders.mean():.2f}")
        else:
            st.info("Insufficient data to perform T-test for gender differences in order value.")
    else:
        st.info("Required columns for gender differences in order value test are missing.")

    # Test 4: Payment Method Success Rate Test
    st.subheader("4. Payment Method Success Rate Test")
    st.markdown("**Hypotheses:**")
    st.markdown("  - **H0:** Payment failure rates are independent of payment method")
    st.markdown("  - **H1:** Payment failure rates depend on payment method")

    if not df_full.empty and 'payment_method' in df_full.columns and 'payment_status' in df_full.columns:
        payment_contingency = pd.crosstab(df_full['payment_method'], df_full['payment_status'])
        if not payment_contingency.empty and payment_contingency.shape[0] > 1 and payment_contingency.shape[1] > 1:
            chi2_payment, p_value_payment, dof_payment, expected_payment = chi2_contingency(payment_contingency)

            st.write(f"**Chi-square statistic:** {chi2_payment:.4f}")
            st.write(f"**P-value:** {p_value_payment:.6f}")

            if p_value_payment < 0.05:
                st.markdown("**Result:** ✓ REJECT null hypothesis (p < 0.05)")
                st.markdown("**Interpretation:** Payment failure rates ARE significantly associated with payment method choice.")
            else:
                st.markdown("**Result:** ✗ FAIL TO REJECT null hypothesis (p >= 0.05)")
                st.markdown("**Interpretation:** There is no statistically significant association between payment failure rates and payment method choice.")

            payment_failure_rates = df_full.groupby('payment_method')['payment_status'].apply(
                lambda x: (x == 'Failed').sum() / len(x) * 100
            ).round(2)

            st.markdown("**Failure Rates by Payment Method (Overall Data):**")
            st.write(payment_failure_rates.to_string())
        else:
            st.info("Insufficient data to perform Chi-square test for payment method success rates.")
    else:
        st.info("Required columns for payment method success rates test are missing.")


    # Test 5: Weekend vs Weekday Sales Test
    st.subheader("5. Weekend vs Weekday Order Values")
    st.markdown("**Hypotheses:**")
    st.markdown("  - **H0:** Mean order value is the same for weekends and weekdays")
    st.markdown("  - **H1:** Mean order value differs between weekends and weekdays")

    if not df_full.empty and 'is_weekend' in df_full.columns and 'total_price_usd' in df_full.columns:
        weekend_orders = df_full[df_full['is_weekend'] == 'Yes']['total_price_usd'].dropna()
        weekday_orders = df_full[df_full['is_weekend'] == 'No']['total_price_usd'].dropna()

        if not weekend_orders.empty and not weekday_orders.empty:
            t_stat_weekend, p_value_weekend = ttest_ind(weekend_orders, weekday_orders)

            st.write(f"**T-statistic:** {t_stat_weekend:.4f}")
            st.write(f"**P-value:** {p_value_weekend:.6f}")

            if p_value_weekend < 0.05:
                st.markdown("**Result:** ✓ REJECT null hypothesis (p < 0.05)")
                st.markdown("**Interpretation:** Weekend and weekday orders have significantly different average values.")
            else:
                st.markdown("**Result:** ✗ FAIL TO REJECT null hypothesis (p >= 0.05)")
                st.markdown("**Interpretation:** There is no statistically significant difference in average order values between weekends and weekdays.")

            st.markdown("**Order Value Statistics (Overall Data):**")
            st.write(f"  - Weekend - Mean: ${weekend_orders.mean():.2f}")
            st.write(f"  - Weekday - Mean: ${weekday_orders.mean():.2f}")
        else:
            st.info("Insufficient data to perform T-test for weekend vs weekday order values.")
    else:
        st.info("Required columns for weekend vs weekday order values test are missing.")

def section_insights_recommendations():
    st.header("💡 Key Business Insights & Recommendations")

    st.subheader("Summary of Key Findings")
    st.markdown("""
    #### 1️⃣ Revenue & Profitability Insights
    - **Total Revenue**: Over the analysis period, the business generated substantial revenue across all customer segments.
    - **Profit Margins**: Overall profit margins are consistent across customer segments, but significantly affected by discounts.
    - **Top Performing Categories**: Electronics, Home, and Sports categories dominate revenue generation.
    - **Geographic Performance**: Netherlands, Italy, and Belgium are the top revenue-generating countries.

    #### 2️⃣ Customer Behavior Insights
    - **Customer Segmentation**: RFM analysis provides granular segments for targeted marketing.
    - **Loyalty Impact**: Customer loyalty scores do not show a strong correlation with overall metrics in this dataset, suggesting other factors are more influential.
    - **Customer Lifetime Value**: Average CLV varies by customer segment, indicating potential for differentiation.
    - **Age Demographics**: Different age groups show varying purchasing behaviors and loyalty scores.
    - **Gender Patterns**: No statistically significant difference in average order values between genders.

    #### 3️⃣ Discount Strategy Insights
    - **Discount Effectiveness**: Orders with discounts show significantly lower profit margins compared to non-discounted orders (p < 0.001).
    - **Optimal Discount Range**: Deeper analysis is needed to identify optimal discount levels that balance volume and margin.
    - **Discount Overuse**: Caution should be exercised with heavy discounting as it substantially erodes profitability.

    #### 4️⃣ Operational Insights
    - **Return Rates**: Overall return rate is around 10%, with 'Wrong Item' and 'Defective Product' being major reasons.
    - **High-Risk Categories**: Specific product categories show higher return rates.
    - **Delivery Performance**: Delivery success rates are high, but failed deliveries and associated costs need monitoring.
    - **Payment Issues**: Payment failure rates are low and not significantly associated with payment method.
    - **Fraud Risk**: Fraud risk scores show slight variations across payment methods, indicating potential areas for enhanced monitoring.

    #### 5️⃣ Marketing Channel Performance
    - **Channel Effectiveness**: Campaign sources like Affiliate and Email generate significant revenue.
    - **Traffic Quality**: Traffic sources show varied average order values and order counts.
    - **Device Preferences**: Revenue contributions are spread across device types, with differences in cart abandonment rates.

    #### 6️⃣ Temporal Patterns
    - **Seasonality**: Clear monthly and daily patterns exist, allowing for strategic planning (e.g., peak ordering hours, consistent revenue months).
    """)

    st.subheader("📊 Statistical Significance Summary")
    st.markdown("""
    All major findings were validated using appropriate statistical tests, mostly on the overall (unfiltered) dataset:
    - **ANOVA (Profit Margins by Customer Segment)**: P-value was **0.541**, indicating NO statistically significant difference in mean profit margins across customer segments (Fail to Reject H0).
    - **T-test (Discount Impact on Profit Margin)**: P-value was **0.000**, indicating a statistically significant difference in mean profit margins between discounted and non-discounted orders (Reject H0). Discounted orders have significantly lower margins.
    - **T-test (Gender Differences in Order Value)**: P-value was **0.916**, indicating NO statistically significant difference in mean order value between males and females (Fail to Reject H0).
    - **Chi-square (Payment Method Success Rates)**: P-value was **0.508**, indicating NO statistically significant association between payment failure rates and payment method choice (Fail to Reject H0).
    - **T-test (Weekend vs Weekday Order Values)**: P-value was **0.859**, indicating NO statistically significant difference in mean order values between weekends and weekdays (Fail to Reject H0).
    """)

    st.subheader("🎯 Strategic Recommendations")
    st.markdown("""
    #### Revenue Optimization
    1. **Focus on High-Value Segments**: Develop targeted retention programs for VIP and Premium customers, leveraging CLV insights.
    2. **Category Optimization**: Double down on high-margin and high-revenue categories (e.g., Electronics) while analyzing underperforming ones.
    3. **Geographic Expansion**: Invest in markets with high growth potential, identified from country-level revenue analysis.

    #### Pricing & Discount Strategy
    1. **Refine Discounting**: Implement data-driven dynamic pricing, shifting from blanket discounts to targeted promotions to protect profit margins.
    2. **A/B Test Discount Levels**: Conduct controlled experiments to find optimal discount percentages that maximize sales volume without excessively eroding profits.

    #### Customer Experience
    1. **Address Return Root Causes**: Investigate and address top return reasons like 'Wrong Item' and 'Defective Product' through improved product descriptions and quality control.
    2. **Enhance Loyalty Programs**: Develop tailored engagement initiatives for at-risk customer segments identified by RFM analysis.

    #### Operational Excellence
    1. **Monitor Shipping Performance**: Continuously optimize shipping methods and warehouse logistics to improve delivery times and reduce costs.
    2. **Strengthen Fraud Prevention**: Enhance screening mechanisms, especially for orders flagged with higher fraud risk scores.

    #### Marketing Efficiency
    1. **Channel Reallocation**: Increase budget for high-ROI marketing channels (e.g., Email, Google Ads) and optimize underperforming ones.
    2. **Mobile Optimization**: Improve mobile user experience and checkout flow to capture the large volume of mobile orders.
    3. **Cart Abandonment Recovery**: Implement targeted retargeting campaigns for abandoned carts, particularly for device types with high abandonment rates.
    """)

    st.subheader("⚠️ Key Risk Areas")
    st.markdown("""
    - **High Return Rates**: Can significantly impact profitability and customer satisfaction.
    - **Over-Discounting**: Leads to margin erosion if not strategically managed.
    - **Payment/Delivery Failures**: Can result in lost revenue and negative customer experiences.
    - **Fraud Risk**: Potential financial losses from fraudulent transactions.
    """)

    st.subheader("📈 Next Steps & Advanced Analytics")
    st.markdown("""
    1. **Predictive Modeling**: Develop models for customer churn, return likelihood, and fraud detection.
    2. **Cohort Analysis**: Conduct in-depth analysis of customer cohorts to understand long-term behavior and value.
    3. **Dynamic Pricing Algorithms**: Implement machine learning models to optimize pricing in real-time.
    4. **Inventory Forecasting**: Utilize time-series models for more accurate demand prediction.
    5. **Advanced Segmentation**: Apply clustering algorithms for even more granular customer segments.
    """)

    st.subheader("📝 Data Quality Notes & Limitations")
    st.markdown("""
    #### Data Quality Notes
    - **Completeness**: Dataset is largely complete, with expected missing patterns for fields like `return_reason`.
    - **Consistency**: Financial calculations and date components are validated and consistent.
    - **Temporal Coverage**: Provides robust seasonal insights over a two-year period.

    #### Limitations
    - **External Factors**: Analysis does not account for macroeconomic conditions, competitor actions, or market trends.
    - **Attribution**: Marketing attribution is single-touch; a multi-touch attribution model could offer deeper insights.
    - **Causality**: Correlations identified do not necessarily imply causation.
    """)

    st.markdown("""
    --- 
    ## 🎉 Conclusion
    This comprehensive exploratory data analysis of the e-commerce dataset has uncovered actionable insights across revenue, customer behavior, operations, and marketing performance. The analysis employed rigorous statistical methods to validate findings and ensure business recommendations are data-driven.

    **Key Takeaways:**
    1. **Customer Segmentation Matters**: RFM and existing segments provide a framework for targeted strategies.
    2. **Discount Strategy Needs Refinement**: Current discounting practices impact profitability, necessitating a more surgical approach.
    3. **Operational Excellence is Critical**: Addressing returns, fraud, and delivery issues can unlock significant value.
    4. **Marketing ROI Varies Widely**: Channel performance differs substantially, suggesting opportunities for budget reallocation.
    5. **Data-Driven Culture**: The rich dataset enables continuous optimization through A/B testing and predictive analytics.

    **Final Recommendation:** Implement a phased approach starting with quick wins (discount optimization, fraud prevention) while building capabilities for advanced analytics (predictive modeling, personalization) to drive sustainable competitive advantage.
    """)

# ----------------------------------------------------------------------------
# Main Streamlit App Logic
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Pricing Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 E-Commerce Pricing Analytics Dashboard")
st.markdown("""
This dashboard provides a comprehensive exploratory data analysis of e-commerce transactions, offering insights into revenue, customer behavior, product performance, and operational efficiency.
""")

# Load data
csv_path = r"C:\Users\akram\Downloads\ecommerce_dataset\1m\csv\ecommerce_dataset_+1m.csv"
df = load_and_preprocess_data(csv_path)

st.success("✓ Data loaded and preprocessed successfully!")

# --- Sidebar for Filters ---
st.sidebar.header("⚙️ Filters")

# 1. Date Range Filter
min_date = df['order_date'].min().date()
max_date = df['order_date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    df_filtered = df[(df['order_date'] >= start_date) & (df['order_date'] <= end_date)].copy()
else:
    df_filtered = df.copy()

# 2. Customer Segment Filter
selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=df_filtered['customer_segment'].unique().tolist(),
    default=df_filtered['customer_segment'].unique().tolist()
)
df_filtered = df_filtered[df_filtered['customer_segment'].isin(selected_segments)]

# 3. Product Category Filter
selected_categories = st.sidebar.multiselect(
    "Product Category",
    options=df_filtered['category'].unique().tolist(),
    default=df_filtered['category'].unique().tolist()
)
df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]

# 4. Country Filter
selected_countries = st.sidebar.multiselect(
    "Country",
    options=df_filtered['country'].unique().tolist(),
    default=df_filtered['country'].unique().tolist()
)
df_filtered = df_filtered[df_filtered['country'].isin(selected_countries)]

# 5. Gender Filter
selected_genders = st.sidebar.multiselect(
    "Gender",
    options=df_filtered['gender'].unique().tolist(),
    default=df_filtered['gender'].unique().tolist()
)
df_filtered = df_filtered[df_filtered['gender'].isin(selected_genders)]

# 6. Discount Applied Filter
discount_choice = st.sidebar.radio(
    "Discount Applied?",
    options=['All', 'Yes', 'No'],
    index=0 # Default to 'All'
)

if discount_choice == 'Yes':
    df_filtered = df_filtered[df_filtered['has_discount'] == 1]
elif discount_choice == 'No':
    df_filtered = df_filtered[df_filtered['has_discount'] == 0]


# Optional: Display raw data sample and info
if st.checkbox("Show Raw Data Sample (Filtered)"):
    st.subheader("Raw Data Sample (Filtered)")
    st.write(df_filtered.head())

if st.checkbox("Show Data Info (Filtered)"):
    st.subheader("Data Information (Filtered)")
    # Redirect info to a string buffer to display in Streamlit
    import io
    buffer = io.StringIO()
    df_filtered.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# Display Key Performance Indicators (KPIs) - using filtered data
st.subheader("🚀 Key Performance Indicators")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Revenue", f"${df_filtered['total_price_usd'].sum():,.2f}")
with col2:
    st.metric("Total Profit", f"${df_filtered['profit_usd'].sum():,.2f}")
with col3:
    st.metric("Avg Order Value", f"${df_filtered['total_price_usd'].mean():,.2f}")
with col4:
    total_orders_filtered = df_filtered['order_id'].nunique()
    returned_orders_filtered = df_filtered[df_filtered['order_status'] == 'Returned']['order_id'].nunique()
    overall_return_rate_filtered = (returned_orders_filtered / total_orders_filtered) * 100 if total_orders_filtered > 0 else 0
    st.metric("Overall Return Rate", f"{overall_return_rate_filtered:,.2f}%")
with col5:
    st.metric("Total Customers", f"{df_filtered['customer_id'].nunique():,}")
with col6:
    st.metric("Total Orders", f"{df_filtered['order_id'].nunique():,}")


st.sidebar.markdown("--- ")
st.sidebar.subheader("Dashboard Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    [
        "Revenue & Profitability",
        "Customer Behavior & Segmentation",
        "Product Performance Deep Dive",
        "Pricing & Discount Strategy",
        "Operational Metrics",
        "Marketing Channel Performance",
        "Statistical Hypothesis Testing",
        "Key Business Insights & Recommendations"
    ]
)

# --- Display selected section ---
if page_selection == "Revenue & Profitability":
    section_revenue_profitability(df_filtered)
elif page_selection == "Customer Behavior & Segmentation":
    section_customer_behavior(df_filtered)
elif page_selection == "Product Performance Deep Dive":
    section_product_performance(df_filtered)
elif page_selection == "Pricing & Discount Strategy":
    section_pricing_discount(df_filtered)
elif page_selection == "Operational Metrics":
    section_operational_metrics(df_filtered)
elif page_selection == "Marketing Channel Performance":
    section_marketing_performance(df_filtered)
elif page_selection == "Statistical Hypothesis Testing":
    section_statistical_tests(df) # Statistical tests usually run on the full dataset for generalizability
elif page_selection == "Key Business Insights & Recommendations":
    section_insights_recommendations()
