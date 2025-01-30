# ======================
# Streamlit App for Hotel Booking Analysis Dashboard
# ======================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import joblib

# Set Streamlit page configuration
st.set_page_config(page_title="Hotel Booking Analysis Dashboard", layout="wide")

# Title and Introduction
st.title("Hotel Booking Analysis Dashboard")
st.markdown("""
This dashboard provides an interactive view of hotel booking data, including cancellation trends, 
key insights, and actionable recommendations. Use the sidebar to navigate between sections.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Overview", "Visual Analytics"])

# ======================
# 1. DATA PROCESSING
# ======================

# Load data
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data():
    raw_df = pd.read_csv('hotel_booking.csv')
    
    # Data validation
    required_columns = ['hotel', 'is_canceled', 'lead_time', 'adults']
    missing = [col for col in required_columns if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Clean data
    raw_df['children'] = raw_df['children'].fillna(0)
    raw_df['country'] = raw_df['country'].fillna('Others')
    raw_df['agent'] = raw_df['agent'].fillna(0)
    raw_df['company'] = raw_df['company'].fillna(0)
    clean_df = raw_df[raw_df['adults'] > 0]
    
    # Feature engineering
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    clean_df['arrival_date_month'] = pd.Categorical(
        clean_df['arrival_date_month'], 
        categories=months, 
        ordered=True
    )
    clean_df['total_nights'] = clean_df['stays_in_weekend_nights'] + clean_df['stays_in_week_nights']
    clean_df['arrival_year_month'] = clean_df['arrival_date_year'].astype(str) + '-' + \
        clean_df['arrival_date_month'].str[:3]
    clean_df['arrival_month_num'] = clean_df['arrival_date_month'].cat.codes + 1
    clean_df['special_requests_flag'] = (clean_df['total_of_special_requests'] > 0).astype(int)
    
    return clean_df

# Load the dataset
clean_df = load_data()

# ======================
# 2. Overview Section
# ======================
if section == "Overview":
    st.header("Project Overview")
    st.markdown("""
    **Business Problem:**  
    Hotels face high cancellation rates, leading to revenue loss and underutilized rooms.  
    This project analyzes hotel booking data to identify cancellation trends and provide actionable business insights.

    **Dataset Details:**  
    - Contains booking information for City Hotel and Resort Hotel (2015-2017).  
    - Includes features like lead time, arrival date, number of guests, special requests, and more.  

    **Key Insights:**  
    """)

    # Display dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(clean_df.head())

    # Display Business Insights
    current_median_wait = clean_df['days_in_waiting_list'].median()
    dynamic_pricing_candidates = len(clean_df[
        (clean_df['lead_time'] > 90) & 
        (clean_df['adr'] > clean_df.groupby('hotel')['adr'].transform('median'))
    ])
    loyalty_prospects = clean_df[
        (clean_df['total_of_special_requests'] >= 2) & 
        (clean_df['previous_cancellations'] == 0)
    ].shape[0]

    st.subheader("Key Metrics")
    st.metric("Current Median Waitlist Days", f"{current_median_wait:.1f}")
    st.metric("Dynamic Pricing Candidates", f"{dynamic_pricing_candidates} bookings")
    st.metric("Loyalty Program Prospects", f"{loyalty_prospects} guests")

    st.subheader("Recommendations")
    st.markdown("""
    1. **Dynamic Pricing Strategy:** Implement dynamic pricing for bookings with long lead times and higher ADR to optimize revenue.  
    2. **Loyalty Program Targeting:** Focus loyalty program efforts on customers who frequently make special requests and have no history of cancellations.  
    3. **Continuous Monitoring:** Regularly monitor cancellation trends and adjust strategies accordingly to maintain optimal revenue generation.
    """)

# ======================
# 3. Visual Analytics Section
# ======================
elif section == "Visual Analytics":
    st.header("Interactive Visual Analytics")
    st.markdown("""
    Explore key trends and patterns in the hotel booking data through interactive visualizations.
    Use the filters in the sidebar to customize your analysis.
    """)

    # Sidebar Filters
    st.sidebar.subheader("Filter Options")

    # Hotel Type Filter
    selected_hotel = st.sidebar.selectbox(
        "Select Hotel Type", 
        options=["All"] + list(clean_df['hotel'].unique())
    )

    # Year Filter
    selected_year = st.sidebar.selectbox(
        "Select Year", 
        options=["All"] + list(clean_df['arrival_date_year'].unique())
    )

    # Month Filter
    selected_month = st.sidebar.selectbox(
        "Select Month", 
        options=["All"] + list(clean_df['arrival_date_month'].unique())
    )

    # Lead Time Slider
    lead_time_range = st.sidebar.slider(
        "Select Lead Time Range (Days)", 
        min_value=int(clean_df['lead_time'].min()), 
        max_value=int(clean_df['lead_time'].max()), 
        value=(int(clean_df['lead_time'].min()), int(clean_df['lead_time'].max()))
    )

    # Apply Filters
    filtered_df = clean_df.copy()
    if selected_hotel != "All":
        filtered_df = filtered_df[filtered_df['hotel'] == selected_hotel]
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['arrival_date_year'] == selected_year]
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df['arrival_date_month'] == selected_month]
    filtered_df = filtered_df[
        (filtered_df['lead_time'] >= lead_time_range[0]) & 
        (filtered_df['lead_time'] <= lead_time_range[1])
    ]

    # Create a compact grid layout for visualizations
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))  # 4 rows, 3 columns
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot 1: Overall Cancellation Rate
    filtered_df['is_canceled'].value_counts(normalize=True).plot.pie(
        autopct='%1.1f%%', colors=sns.color_palette("viridis"),
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, ax=axes[0]
    )
    axes[0].set_title("Overall Cancellation Rate", fontweight='bold')

    # Plot 2: Cancellation by Hotel Type
    sns.barplot(x='hotel', y='is_canceled', data=filtered_df, errorbar=None,
                palette="viridis", order=['City Hotel', 'Resort Hotel'], ax=axes[1])
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].set_title("Cancellation by Hotel Type", fontweight='bold')

    # Plot 3: Lead Time Distribution
    sns.boxplot(x='is_canceled', y='lead_time', data=filtered_df, palette="viridis", showfliers=False, ax=axes[2])
    axes[2].set_title("Lead Time Distribution", fontweight='bold')
    axes[2].set_xticklabels(['Not Canceled', 'Canceled'])

    # Plot 4: Monthly Cancellation Trends
    monthly_data = filtered_df.groupby(['arrival_date_month', 'hotel'])['is_canceled'].mean().reset_index()
    sns.lineplot(x='arrival_date_month', y='is_canceled', hue='hotel',
                 data=monthly_data, marker='o', palette="viridis", linewidth=2.5, sort=False, ax=axes[3])
    axes[3].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[3].set_title("Monthly Cancellation Trends", fontweight='bold')

    # Plot 5: Cancellation by Market Segment
    segment_cancel = filtered_df.groupby('market_segment')['is_canceled'].mean().sort_values()
    sns.barplot(y=segment_cancel.index, x=segment_cancel.values, palette="viridis", ax=axes[4])
    axes[4].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[4].set_title("Cancellation by Market Segment", fontweight='bold')

    # Plot 6: Cancellation by Deposit Type
    deposit_cancel = filtered_df.groupby('deposit_type')['is_canceled'].mean().sort_values()
    sns.barplot(y=deposit_cancel.index, x=deposit_cancel.values, palette="viridis", ax=axes[5])
    axes[5].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[5].set_title("Cancellation by Deposit Type", fontweight='bold')

    # Plot 7: Repeat vs New Guests
    sns.countplot(x='is_repeated_guest', hue='is_canceled', data=filtered_df, palette="viridis", ax=axes[6])
    axes[6].set_title("Repeat vs New Guests", fontweight='bold')
    axes[6].set_xticklabels(['No', 'Yes'])

    # Plot 8: Feature Correlation Matrix
    corr_matrix = filtered_df[['lead_time', 'adr', 'total_nights', 'previous_cancellations',
                               'days_in_waiting_list', 'is_canceled']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', center=0, ax=axes[7])
    axes[7].set_title("Feature Correlation Matrix", fontweight='bold')

    # Plot 9: ADR Distribution
    sns.boxplot(x='is_canceled', y='adr', data=filtered_df, palette="viridis", showfliers=False, ax=axes[8])
    axes[8].set_xticklabels(['Not Canceled', 'Canceled'])
    axes[8].set_title("ADR Distribution", fontweight='bold')

    # Plot 10: Special Requests Impact
    sns.barplot(x='total_of_special_requests', y='is_canceled', data=filtered_df,
                errorbar=None, palette="viridis", ax=axes[9])
    axes[9].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[9].set_title("Special Requests Impact", fontweight='bold')

    # Plot 11: Cancellation Trends by Booking Year-Month
    year_month_data = filtered_df.groupby(['arrival_year_month', 'hotel'])['is_canceled'].mean().reset_index()
    sns.lineplot(x='arrival_year_month', y='is_canceled', hue='hotel',
                 data=year_month_data, palette="viridis", linewidth=2.5, ax=axes[10])
    axes[10].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[10].set_title("Cancellation Trends by Booking Year-Month", fontweight='bold')
    axes[10].tick_params(axis='x', rotation=45)

    # Remove unused subplot
    axes[11].axis('off')

    # Adjust layout and display the figure
    plt.tight_layout()
    st.pyplot(fig)
