import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    client = Groq(api_key=groq_api_key)
else:
    client = None

st.set_page_config(
    page_title="Automated EDA Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automated EDA Tool")

st.write(
    "Upload a CSV dataset to automatically explore the data."
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # =====================================================
    # DATASET STATISTICS
    # =====================================================

    st.subheader("📊 Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )

    col4.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

    # =====================================================
    # COLUMN INFORMATION
    # =====================================================

    st.subheader("🔎 Column Information")

    information = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        information,
        use_container_width=True
    )

    # =====================================================
    # NUMERICAL STATISTICS
    # =====================================================

    st.subheader("📈 Numerical Statistics")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:

        st.dataframe(
            df[numeric_columns].describe(),
            use_container_width=True
        )

    else:

        st.info("No numerical columns found.")

    # =====================================================
    # VISUALIZATIONS
    # =====================================================

    st.header("📊 Data Visualizations")

    # -----------------------------------------------------
    # NUMERICAL VISUALIZATIONS
    # -----------------------------------------------------

    if len(numeric_columns) > 0:

        st.subheader("📈 Numerical Distributions")

        selected_numeric = st.selectbox(
            "Select a numerical column",
            numeric_columns
        )

        fig, ax = plt.subplots()

        sns.histplot(
            df[selected_numeric].dropna(),
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Distribution of {selected_numeric}"
        )

        ax.set_xlabel(selected_numeric)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

        # -------------------------------------------------
        # BOX PLOT
        # -------------------------------------------------

        st.subheader("📦 Outlier Analysis")

        fig, ax = plt.subplots()

        sns.boxplot(
            x=df[selected_numeric],
            ax=ax
        )

        ax.set_title(
            f"Box Plot of {selected_numeric}"
        )

        st.pyplot(fig)

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    if len(numeric_columns) >= 2:

        st.subheader("🔥 Correlation Heatmap")

        correlation = df[numeric_columns].corr()

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        ax.set_title(
            "Correlation Between Numerical Variables"
        )

        st.pyplot(fig)

    # =====================================================
    # CATEGORICAL DATA
    # =====================================================

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    if len(categorical_columns) > 0:

        st.subheader("📊 Categorical Data")

        selected_category = st.selectbox(
            "Select a categorical column",
            categorical_columns
        )

        category_counts = (
            df[selected_category]
            .value_counts()
            .head(10)
        )

        fig, ax = plt.subplots()

        category_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            f"Top Categories in {selected_category}"
        )

        ax.set_xlabel(selected_category)
        ax.set_ylabel("Count")

        plt.xticks(rotation=45)

        st.pyplot(fig)

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    st.header("📝 EDA Summary")

    st.write(
        f"""
        The dataset contains **{df.shape[0]} rows**
        and **{df.shape[1]} columns**.

        There are **{int(df.isna().sum().sum())} missing values**
        and **{int(df.duplicated().sum())} duplicate rows**.
        """
    )