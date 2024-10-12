import streamlit as st
import pandas as pd
import random
import plotly.graph_objects as go

data = pd.read_csv('Data.csv')
# Set page config (this must be the first Streamlit command)
st.set_page_config(layout="wide")
st.sidebar.title("Filtering Options")

# Load the DataFrame (replace with your actual dataset path)
df = pd.read_csv('resume_text.csv')

# Main UI - Display the table if there is data
st.title("All Candidates")

# Initialize session state to manage which profile is being viewed
if 'view_profile' not in st.session_state:
    st.session_state.view_profile = False
    st.session_state.current_profile_id = None

# Define function to go back to the main list
def back_to_list():
    st.session_state.view_profile = False
    st.session_state.current_profile_id = None

# Function to create a circular bar plot
def create_circular_bar(score, label, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': label},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5], 'color': 'lightgray'},
                {'range': [5, 10], 'color': 'lightblue'}],
        },
    ))
    fig.update_layout(autosize=False, width=200, height=200, margin=dict(l=20, r=20, t=50, b=50))
    return fig

if df.empty:
    st.write("No candidates available.")
else:
    # Check if we are in profile view or main list view
    if not st.session_state.view_profile:
        # Create a new DataFrame to display only the index as ID
        df_display = pd.DataFrame({'ID': df.index})

        # Limit to the top 10 candidates
        max_display = 10
        for i in range(min(max_display, len(df_display))):
            # Create three columns: ID, CV Score, View Profile button
            col1, col2, col3 = st.columns([3, 1, 1])  # Adjust the column width ratio as needed

            with col1:
                # Display the ID
                st.write(f"ID: {df_display['ID'][i]}")

            with col2:
                # Generate and display a random CV score between 0 and 10
                cv_score = round(random.uniform(0, 10), 2)
                st.write(f"CV Score: {cv_score}")

            with col3:
                # Add a button to view profile
                if st.button(f"View Profile for ID {df_display['ID'][i]}"):
                    st.session_state.view_profile = True
                    st.session_state.current_profile_id = df_display['ID'][i]

    else:
        # Create a centered layout for score columns
        score_col1, score_col2, score_col3, score_col4, score_col5 = st.columns(5)

        # Display scores
        impact_score = data['Quantify impact_score'][st.session_state.current_profile_id]*100
        brevity_score = data['Repeated_Score'][st.session_state.current_profile_id]*100
        style_score = 0
        sections_score = 0
        credibility_score = data['CreditScore'][st.session_state.current_profile_id]
        import matplotlib.colors as mcolors
        def score_to_color(score):
            # Normalize score to be between 0 and 1
            normalized_score = score / 100.0
            # Calculate color components
            red = 1 - normalized_score  # 1 for red when score is 0, 0 for red when score is 100
            green = normalized_score     # 1 for green when score is 100, 0 for green when score is 0
            return mcolors.to_hex((red, green, 0))  # Assuming RGB, no blue

        # Show each score in its column
        with score_col1:
            st.plotly_chart(create_circular_bar(impact_score, "Impact", score_to_color(impact_score)), use_container_width=True)
        with score_col2:
            st.plotly_chart(create_circular_bar(brevity_score, "Brevity", score_to_color(brevity_score)), use_container_width=True)
        with score_col3:
            st.plotly_chart(create_circular_bar(style_score, "Style",score_to_color(style_score)), use_container_width=True)
        with score_col4:
            st.plotly_chart(create_circular_bar(sections_score, "Sections", score_to_color(sections_score)), use_container_width=True)
        with score_col5:
            st.plotly_chart(create_circular_bar(credibility_score, "Credibility", score_to_color(credibility_score)), use_container_width=True)
    
        # Calculate and display overall score
        overall_score = round((impact_score + brevity_score + style_score + sections_score + credibility_score) / 5, 2)
 
        def create_overall_score_gauge(score):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Overall Score"},
                gauge={
                    'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': 'blue'},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 5], 'color': 'lightgray'},
                        {'range': [5, 10], 'color': 'lightblue'}],
                },
            ))
            fig.update_layout(autosize=False, width=300, height=300)
            return fig

        # Calculate overall score
        overall_score = round((impact_score + brevity_score + style_score + sections_score + credibility_score) / 5, 2)

        # Create three columns
        col1, col2, col3 = st.columns(3)

        def create_stacked_horizontal_bar_plot(hard_data, soft_data, title):
            fig = go.Figure()

            # Add hard skills to the plot
            for skill, score in hard_data.items():
                fig.add_trace(go.Bar(
                    y=[skill],
                    x=[score],
                    orientation='h',
                    name=f'Hard Skill: {skill}',
                    hoverinfo='x+y',
                    marker=dict(color='lightblue')
                ))

            # Add soft skills to the plot
            for skill, score in soft_data.items():
                fig.add_trace(go.Bar(
                    y=[skill],
                    x=[score],
                    orientation='h',
                    name=f'Soft Skill: {skill}',
                    hoverinfo='x+y',
                    marker=dict(color='lightgreen')
                ))

            fig.update_layout(
                title=title,
                xaxis_title='Score',
                yaxis_title='Skills',
                xaxis=dict(range=[0, 10]),
                barmode='stack',  # Stack bars on top of each other
                template='plotly_white'
            )
            return fig
        overall_score = round((impact_score + brevity_score + style_score + sections_score + credibility_score) / 5, 2)
        df2 = pd.read_csv('Data2.csv')

        # Create three columns
        col1, col2, col3 = st.columns(3)

        # Plot hard and soft skills in col1
        with col1:
            st.markdown("<h5 style='text-align: center;'>Skills Overview</h5>", unsafe_allow_html=True)
    


        # Keep col2 blank for future content
        with col2:
            st.write("")  # Placeholder for potential future content

        # Display overall score with a styled background in col3
        with col3:

            # Display the overall score as a circular gauge in col3 (assuming create_overall_score_gauge function is defined)
            st.plotly_chart(create_overall_score_gauge(overall_score), use_container_width=True)



        # Add a back button to go back to the main list
        st.button("Back to list", on_click=back_to_list)
