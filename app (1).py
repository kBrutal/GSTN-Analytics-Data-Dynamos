import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pickle
import xgboost as xgb

# Inverted color theme settings
plt.style.use('dark_background')  # Apply a dark background to all plots
sns.set_style('darkgrid')  # Seaborn style with a dark grid background

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        /* Set background color to light blue with a white gradient */
        body {
            background: linear-gradient(to bottom, lightblue, white);
            color: white; /* Ensure text color is readable */
        }
        h1, h2, h3, h4, h5, h6 {
            color: white; /* Set header colors to white for contrast */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    df = pd.read_csv("X_Train_Data_Input.csv")
    target = pd.read_csv("Y_Train_Data_Target.csv")
    df['ID'] = df['ID'].astype(str)
    df = pd.merge(df, target, on='ID')
    return df


def plot_nan_percentage(df):
    nan_percentage = calculate_nan_percentage(
        df.drop(['ID', 'target'], axis=1))

    dropped_columns = pd.DataFrame({
        'Column': nan_percentage[nan_percentage > 30].index,
        'NaN Percentage': nan_percentage[nan_percentage > 30].values,
        'Status': 'Dropped'
    })
    kept_columns = pd.DataFrame({
        'Column': nan_percentage[nan_percentage <= 30].index,
        'NaN Percentage': nan_percentage[nan_percentage <= 30].values,
        'Status': 'Kept'
    })

    combined_df = pd.concat([dropped_columns, kept_columns])

    fig, ax = plt.subplots(figsize=(22, 8), dpi=200)

    # Set the background color for the figure and axis
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Adjust colors for compatibility with dark mode
    sns.barplot(x='Column', y='NaN Percentage', hue='Status',
                data=combined_df, ax=ax, palette="bwr")

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', color='white')

    ax.set_title('NaN Percentage by Column', color='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, color='white')
    ax.set_yticklabels(ax.get_yticks(), color='white')

    # Set the legend text color to black
    ax.legend(facecolor='black', framealpha=0.3, edgecolor='black',
              loc='upper right', labelcolor='white', fontsize=15)

    return fig


def plot_class_distribution(df):
    distribution = class_distribution(df)
    fig, ax = plt.subplots(figsize=(15, 5))

    # Set the background color for the figure and axis
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Adjust colors for compatibility with dark mode
    sns.barplot(x=distribution.index, y=distribution.values,
                ax=ax, palette="muted")

    ax.set_title('Class Distribution (Percentage)', color='white')
    ax.set_xlabel('Class', color='white')  # Add x label
    ax.set_ylabel('Percentage', color='white')  # Add y label
    ax.set_xticklabels(ax.get_xticklabels(), color='white')
    ax.set_yticklabels(ax.get_yticks(), color='white')

    # Set the legend text color to black
    ax.legend(facecolor='black', framealpha=0.3,
              edgecolor='black', labelcolor='black')

    return fig


def calculate_nan_percentage(df):
    nan_percentage = df.isna().mean() * 100
    return nan_percentage


def class_distribution(df):
    return df['target'].value_counts(normalize=True) * 100


def generate_boxen_plots(df, column):
    if column != 'target' and df[column].dtype != 'object':
        fig, ax = plt.subplots(figsize=(15, 5))

        # Set the background color for the figure and axis
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Adjust colors for compatibility with dark mode
        sns.boxenplot(x=df[column], ax=ax, palette="muted")

        ax.set_title(f'Boxen plot of {column}', color='white')
        ax.set_xticklabels(ax.get_xticklabels(), color='white')
        ax.set_yticklabels(ax.get_yticks(), color='white')
        ax.set_xlabel('Value', color='white')
        ax.set_ylabel('Count', color='white')

        return fig


def generate_histogram(df, column):
    if column != 'target' and df[column].dtype != 'object':
        fig, ax = plt.subplots(figsize=(12, 5))

        # Set the background color for the figure and axis
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Adjust colors for compatibility with dark mode
        sns.histplot(df, x=column, hue='target',
                     kde=True, ax=ax, palette="colorblind")

        ax.set_title(f'Histogram of {column} with target hue', color='white')
        ax.set_xticklabels(ax.get_xticklabels(), color='white')
        ax.set_yticklabels(ax.get_yticks(), color='white')
        ax.set_xlabel('Value', color='white')
        ax.set_ylabel('Count', color='white')

        # Customize the legend
        legend = ax.get_legend()
        # Keep the title as it is, we'll customize its color below
        legend.set_title("Target")
        legend.get_frame().set_facecolor('black')
        legend.get_frame().set_edgecolor('black')

        # Customize legend title and text color
        for text in legend.get_texts():
            text.set_color('white')  # Set legend text color to white
        legend.get_title().set_color('white')  # Set legend title color to white

        return fig

def make_donut(size, label, color, value):
    fig, ax = plt.subplots(figsize=(size / 30, size / 30), dpi=10)
    wedges, texts, autotexts = ax.pie([1], labels=[label], colors=[color], autopct=lambda p: f'{value}', startangle=90, wedgeprops=dict(width=0.3))  # Reduced width by a factor of 2

    # Set the background color for the figure and axis
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Adjust text colors for compatibility with dark mode
    for text in texts + autotexts:
        text.set_color('white')

    ax.set_aspect('equal')
    return fig

def score_to_color(score):
    # Normalize score to be between 0 and 1
    normalized_score = score 
    # Calculate color components
    red = 1 - normalized_score  # Red decreases as score increases
    green = normalized_score      # Green increases as score increases
    return mcolors.to_hex((red, green, 0))  # RGB format

def create_circular_bar(score, label, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': label, 'font': {'size': 20, 'color': "white"}},  # Title in white
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "lightblue",  # Set background color to light blue
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 0.5], 'color': 'lightgray'},
                {'range': [0.5, 1], 'color': 'lightblue'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},  # Optional: Threshold line can be customized
                'thickness': 0.75,
                'value': 0.42  # Optional: Example threshold value
            }
        },
        number={'font': {'color': "white", 'size': 20}}  # Number displayed in white
    ))

    fig.update_layout(autosize=False, width=200, height=230, margin=dict(l=20, r=20, t=50, b=50))
    return fig

import matplotlib.colors as mcolors
import plotly.graph_objects as go
def main():
    st.title("GST Prediction App")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Analysis", "Prediction"])

    df = load_data()

    st.sidebar.title("Data Analysis Options")
    analysis_option = st.sidebar.selectbox("Choose an analysis option", [
                                           "None", "NaN Percentage by Column", "Class Distribution (Target Column)", "Boxen Plot for Each Column", "Histogram with Target Hue"])

    if page == "Data Analysis":
        st.header("Data Analysis")
        if analysis_option in ["Boxen Plot for Each Column", "Histogram with Target Hue"]:
            column_option = st.sidebar.selectbox("Choose a column", [
                                                 col for col in df.columns if col != 'target' and df[col].dtype != 'object'])

        if analysis_option == "Boxen Plot for Each Column":
            if column_option:
                fig = generate_boxen_plots(df, column_option)
                st.pyplot(fig)

        elif analysis_option == "Histogram with Target Hue":
            if column_option:
                fig = generate_histogram(df, column_option)
                st.pyplot(fig)

        elif analysis_option == "NaN Percentage by Column":
            fig = plot_nan_percentage(df)
            st.pyplot(fig)

        elif analysis_option == "Class Distribution (Target Column)":
            fig = plot_class_distribution(df)
            st.pyplot(fig)

    else:
        st.title("Prediction")
        st.header("Input Data for Prediction")

        # Create a form for user input
        with st.form(key='prediction_form'):
            user_input = {}
            cols = st.columns(4)  # Create 4 columns for the form layout
            for idx, column in enumerate(df.columns):
                if column != 'ID' and column != 'target':
                    col = cols[idx % 4]  # Distribute inputs across the 4 columns
                    user_input[column] = col.number_input(f"{column}", value=0.0)
            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            
            
            # Load the model from the file
            with open('xgboost_model.pkl', 'rb') as file:
                model = pickle.load(file)

            # Convert user_input to a DataFrame
            input_df = pd.DataFrame([user_input])
            input_df.drop(columns = ['Column14', 'Column9'], inplace = True)
            input_dmatrix = xgb.DMatrix(input_df)

            # Predict the value using the loaded model
            ans = model.predict(input_dmatrix)[0]
            st.title("Prediction:-")

            score_col1, score_col2, score_col3 = st.columns(3)

            with score_col1:
                pass
            with score_col2:
                st.plotly_chart(create_circular_bar(ans, "Label 1 Probability", score_to_color(ans)), use_container_width=True)
            with score_col3:
                pass


if __name__ == "__main__":
    main()
