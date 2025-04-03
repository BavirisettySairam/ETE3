import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import nltk
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime, timedelta
import random
import numpy as np
import seaborn as sns
import os
from PIL import Image, ImageEnhance, ImageFilter
import io

# Set the style for all plots
plt.style.use('seaborn-v0_8')
sns.set_theme()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Set page config
st.set_page_config(
    page_title="INBLOOM '25 Dashboard",
    page_icon="🎭",
    layout="wide"
)

# Cache the dataset generation
@st.cache_data
def generate_dataset():
    # Define the data
    events = [
        "Dance Competition", "Music Festival", "Drama Show", "Art Exhibition",
        "Debate Competition", "Photography Contest", "Fashion Show",
        "Poetry Slam", "Stand-up Comedy", "Cultural Quiz"
    ]
    
    colleges = [
        "St. Xavier's College", "Loyola College", "Christ University",
        "Mount Carmel College", "Presidency College", "Bangalore University",
        "Mysore University", "Manipal University", "VIT University",
        "SRM University"
    ]
    
    states = ["Karnataka", "Tamil Nadu", "Kerala", "Andhra Pradesh", "Telangana"]
    
    # Generate random data
    n_participants = 250
    data = {
        'Participant_ID': range(1, n_participants + 1),
        'Name': [f"Participant_{i}" for i in range(1, n_participants + 1)],
        'College': random.choices(colleges, k=n_participants),
        'State': random.choices(states, k=n_participants),
        'Event': random.choices(events, k=n_participants),
        'Day': random.choices(range(1, 6), k=n_participants),
        'Registration_Time': [datetime.now() - timedelta(days=random.randint(1, 30)) for _ in range(n_participants)],
        'Participation_Status': random.choices(['Completed', 'Scheduled', 'Cancelled'], k=n_participants),
        'Rating': random.choices(range(1, 6), k=n_participants),
        'Feedback': [
            f"Great experience at {event}! The organization was excellent."
            for event in random.choices(events, k=n_participants)
        ]
    }
    
    return pd.DataFrame(data)

# Load the dataset
df = generate_dataset()

# Title and Description
st.title("🎭 INBLOOM '25")
st.markdown("### Inter-College Cultural Events Dashboard")
st.markdown("Insights into participation trends, feedback analysis, and event highlights")

# Sidebar with enhanced design
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: #1E3D59;'>🎭 INBLOOM '25</h2>
        <p style='color: #666;'>Dashboard Navigation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation with icons
    page = st.radio(
        "Select a section",
        ["Generated Data", "Overview", "Participation Analysis", "Feedback Analysis", "Image Gallery", "About"],
        label_visibility="collapsed",
        index=1  # Set Overview as default
    )
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Add quick stats in sidebar
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
        <h4 style='color: #1E3D59;'>Quick Stats</h4>
        <p>👥 Participants: {}</p>
        <p>🎭 Events: {}</p>
        <p>🏫 Colleges: {}</p>
        <p>⭐ Avg Rating: {}</p>
    </div>
    """.format(
        len(df),
        len(df['Event'].unique()),
        len(df['College'].unique()),
        round(df['Rating'].mean(), 1)
    ), unsafe_allow_html=True)
    
    # Add footer in sidebar
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        <p>Built with ❤️</p>
        <p>by Bavirisetty Sairam</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for filters if they don't exist
if 'selected_event' not in st.session_state:
    st.session_state.selected_event = "All"
if 'selected_college' not in st.session_state:
    st.session_state.selected_college = "All"
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = "All"
if 'compare_event1' not in st.session_state:
    st.session_state.compare_event1 = df['Event'].unique()[0]
if 'compare_event2' not in st.session_state:
    st.session_state.compare_event2 = df['Event'].unique()[1]
if 'gallery_event' not in st.session_state:
    st.session_state.gallery_event = "All"
if 'gallery_college' not in st.session_state:
    st.session_state.gallery_college = "All"

# Main content based on navigation
if page == "Overview":
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Participants", len(df))
        st.caption("Total number of registered participants")
    
    with col2:
        st.metric("Total Events", len(df['Event'].unique()))
        st.caption("Number of different events")
    
    with col3:
        st.metric("Total Colleges", len(df['College'].unique()))
        st.caption("Number of participating colleges")
    
    with col4:
        st.metric("Average Rating", round(df['Rating'].mean(), 1))
        st.caption("Overall event rating (1-5)")
    
    # Event participation distribution
    st.subheader("Event Participation Distribution")
    event_counts = df['Event'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=event_counts.index, y=event_counts.values, ax=ax, palette='husl')
    plt.xticks(rotation=45)
    plt.title("Event-wise Participation", pad=20)
    st.pyplot(fig)
    st.caption("Distribution of participants across different events")
    
    # Additional charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Participation Status")
        status_counts = df['Participation_Status'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=sns.color_palette('husl'))
        plt.title("Participation Status Distribution")
        st.pyplot(fig)
        st.caption("Breakdown of participation status across all events")
    
    with col2:
        st.subheader("State-wise Participation")
        state_counts = df['State'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=state_counts.index, y=state_counts.values, ax=ax, palette='husl')
        plt.xticks(rotation=45)
        plt.title("State-wise Participation")
        st.pyplot(fig)
        st.caption("Distribution of participants from different states")

elif page == "Participation Analysis":
    st.header("📈 Participation Analysis")
    
    # Filters with session state
    col1, col2, col3 = st.columns(3)
    with col1:
        options = ["All"] + list(df['Event'].unique())
        selected_event = st.selectbox(
            "Select Event",
            options,
            key="event_select",
            index=options.index(st.session_state.selected_event)
        )
        st.session_state.selected_event = selected_event
    
    with col2:
        options = ["All"] + list(df['College'].unique())
        selected_college = st.selectbox(
            "Select College",
            options,
            key="college_select",
            index=options.index(st.session_state.selected_college)
        )
        st.session_state.selected_college = selected_college
    
    with col3:
        options = ["All"] + list(df['State'].unique())
        selected_state = st.selectbox(
            "Select State",
            options,
            key="state_select",
            index=options.index(st.session_state.selected_state)
        )
        st.session_state.selected_state = selected_state
    
    # Apply filters
    filtered_df = df.copy()
    if st.session_state.selected_event != "All":
        filtered_df = filtered_df[filtered_df['Event'] == st.session_state.selected_event]
    if st.session_state.selected_college != "All":
        filtered_df = filtered_df[filtered_df['College'] == st.session_state.selected_college]
    if st.session_state.selected_state != "All":
        filtered_df = filtered_df[filtered_df['State'] == st.session_state.selected_state]
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Day-wise Participation")
        day_counts = filtered_df['Day'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax, palette='husl')
        plt.title("Day-wise Participation")
        plt.xlabel("Day")
        plt.ylabel("Number of Participants")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Daily participation trends")
    
    with col2:
        st.subheader("College-wise Participation")
        college_counts = filtered_df['College'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=college_counts.index, y=college_counts.values, ax=ax, palette='husl')
        plt.title("College-wise Participation")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Distribution of participants from different colleges")
    
    # Additional charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=filtered_df, x='Rating', bins=5, ax=ax, color='#ff4b4b')
        plt.title("Rating Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Distribution of ratings given by participants")
    
    with col2:
        st.subheader("Participation Status")
        status_counts = filtered_df['Participation_Status'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=sns.color_palette('husl'))
        plt.title("Participation Status Distribution")
        st.pyplot(fig)
        st.caption("Breakdown of participation status for selected filters")

elif page == "Feedback Analysis":
    st.header("💬 Feedback Analysis")
    
    # Define stopwords once for both tabs
    stopwords = set(STOPWORDS)
    stopwords.update(['great', 'experience', 'organization', 'excellent', 'atmosphere', 'amazing'])
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Single Event Analysis", "Event Comparison"])
    
    with tab1:
        # Event selection for feedback with "All" option
        options = ["All"] + list(df['Event'].unique())
        selected_event = st.selectbox("Select Event for Feedback Analysis", options)
        
        # Get feedback based on selection
        if selected_event == "All":
            event_feedback = df['Feedback']
            event_ratings = df['Rating']
        else:
            event_feedback = df[df['Event'] == selected_event]['Feedback']
            event_ratings = df[df['Event'] == selected_event]['Rating']
        
        # Generate word cloud with NLTK stopwords
        st.subheader("Word Cloud of Feedback")
        text = ' '.join(event_feedback)
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            stopwords=stopwords,
            max_words=200,
            relative_scaling=0.6,
            min_font_size=8,
            max_font_size=100,
            prefer_horizontal=0.7,
            collocations=True,
            colormap='viridis'
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.title("Feedback Word Cloud", pad=20)
        st.pyplot(fig)
        st.caption("Most frequent words in feedback, with common words removed")
        
        # Rating distribution
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=event_ratings, bins=5, ax=ax, color='#ff4b4b')
        plt.title(f"Rating Distribution for {selected_event}")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Distribution of ratings given by participants")
        
        # Feedback metrics
        st.subheader("Feedback Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rating", round(event_ratings.mean(), 1))
            st.caption("Mean rating (1-5)")
        with col2:
            st.metric("Total Feedback", len(event_feedback))
            st.caption("Number of feedback entries")
        with col3:
            st.metric("Positive Feedback", round(len(event_ratings[event_ratings >= 4]) / len(event_ratings) * 100, 1))
            st.caption("Percentage of ratings ≥ 4")
    
    with tab2:
        st.subheader("Compare Events")
        
        # Event selection for comparison
        col1, col2 = st.columns(2)
        with col1:
            event1 = st.selectbox("Select First Event", df['Event'].unique(), key="event1")
        with col2:
            event2 = st.selectbox("Select Second Event", df['Event'].unique(), key="event2")
        
        # Get feedback for both events
        feedback1 = df[df['Event'] == event1]['Feedback']
        feedback2 = df[df['Event'] == event2]['Feedback']
        ratings1 = df[df['Event'] == event1]['Rating']
        ratings2 = df[df['Event'] == event2]['Rating']
        
        # Compare word clouds
        st.subheader("Word Cloud Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Word Cloud for {event1}")
            text1 = ' '.join(feedback1)
            wordcloud1 = WordCloud(
                width=400,
                height=300,
                background_color='white',
                stopwords=stopwords,
                max_words=100,
                relative_scaling=0.6,
                min_font_size=8,
                max_font_size=100,
                prefer_horizontal=0.7,
                collocations=True,
                colormap='viridis'
            ).generate(text1)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud1, interpolation='bilinear')
            ax.axis('off')
            plt.title(f"Feedback for {event1}", pad=20)
            st.pyplot(fig)
            st.caption(f"Most frequent words in feedback for {event1}")
        
        with col2:
            st.write(f"Word Cloud for {event2}")
            text2 = ' '.join(feedback2)
            wordcloud2 = WordCloud(
                width=400,
                height=300,
                background_color='white',
                stopwords=stopwords,
                max_words=100,
                relative_scaling=0.6,
                min_font_size=8,
                max_font_size=100,
                prefer_horizontal=0.7,
                collocations=True,
                colormap='viridis'
            ).generate(text2)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud2, interpolation='bilinear')
            ax.axis('off')
            plt.title(f"Feedback for {event2}", pad=20)
            st.pyplot(fig)
            st.caption(f"Most frequent words in feedback for {event2}")
        
        # Compare ratings
        st.subheader("Rating Distribution Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=ratings1, bins=5, alpha=0.5, label=event1, color='blue')
        sns.histplot(data=ratings2, bins=5, alpha=0.5, label=event2, color='red')
        plt.title("Rating Distribution Comparison")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Comparison of rating distributions between the two events")
        
        # Compare metrics
        st.subheader("Event Metrics Comparison")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Average Rating - {event1}", round(ratings1.mean(), 1))
            st.caption("Mean rating")
        with col2:
            st.metric(f"Average Rating - {event2}", round(ratings2.mean(), 1))
            st.caption("Mean rating")
        with col3:
            st.metric(f"Feedback Count - {event1}", len(feedback1))
            st.caption("Total feedback received")
        with col4:
            st.metric(f"Feedback Count - {event2}", len(feedback2))
            st.caption("Total feedback received")
        
        # Additional comparison metrics
        st.subheader("Detailed Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Positive Feedback % - {event1}", 
                     round(len(ratings1[ratings1 >= 4]) / len(ratings1) * 100, 1))
            st.caption("Percentage of ratings ≥ 4")
        with col2:
            st.metric(f"Positive Feedback % - {event2}", 
                     round(len(ratings2[ratings2 >= 4]) / len(ratings2) * 100, 1))
            st.caption("Percentage of ratings ≥ 4")

elif page == "Image Gallery":
    st.header("📸 Image Gallery")
    st.markdown("Upload and process event-related images")
    
    # Create directory for uploaded images if it doesn't exist
    if not os.path.exists("uploaded_images"):
        os.makedirs("uploaded_images")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image using PIL
        image = Image.open(uploaded_file)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, caption="Original Image", use_container_width=True)
        
        # Image processing options
        st.subheader("Image Processing")
        col1, col2 = st.columns(2)
        
        with col1:
            # Brightness adjustment
            brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.1)
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                brightened_image = enhancer.enhance(brightness)
                st.image(brightened_image, caption=f"Brightness: {brightness}", use_container_width=True)
            
            # Contrast adjustment
            contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.1)
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                contrasted_image = enhancer.enhance(contrast)
                st.image(contrasted_image, caption=f"Contrast: {contrast}", use_container_width=True)
        
        with col2:
            # Blur effect
            blur = st.slider("Blur", 0, 10, 0)
            if blur > 0:
                blurred_image = image.filter(ImageFilter.GaussianBlur(blur))
                st.image(blurred_image, caption=f"Blur: {blur}", use_container_width=True)
            
            # Sharpness adjustment
            sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                sharpened_image = enhancer.enhance(sharpness)
                st.image(sharpened_image, caption=f"Sharpness: {sharpness}", use_container_width=True)
        
        # Save processed image
        if st.button("Save Processed Image"):
            # Create a copy of the image with all applied effects
            processed_image = image.copy()
            
            # Apply brightness
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(processed_image)
                processed_image = enhancer.enhance(brightness)
            
            # Apply contrast
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(processed_image)
                processed_image = enhancer.enhance(contrast)
            
            # Apply blur
            if blur > 0:
                processed_image = processed_image.filter(ImageFilter.GaussianBlur(blur))
            
            # Apply sharpness
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(processed_image)
                processed_image = enhancer.enhance(sharpness)
            
            # Save the processed image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"uploaded_images/processed_{timestamp}.jpg"
            processed_image.save(filename)
            st.success(f"Processed image saved as {filename}")

elif page == "Generated Data":
    st.header("📊 Generated Dataset")
    st.markdown("View and explore the complete dataset used in the dashboard")
    
    # Display the full dataset
    st.subheader("Complete Dataset")
    st.dataframe(df, use_container_width=True)
    
    # Dataset Information
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Shape:**")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        
        st.markdown("**Columns:**")
        for col in df.columns:
            st.write(f"- {col}")
    
    with col2:
        st.markdown("**Data Types:**")
        st.write(df.dtypes)
        
        st.markdown("**Missing Values:**")
        st.write(df.isnull().sum())
    
    # Sample Statistics
    st.subheader("Sample Statistics")
    st.write(df.describe())

elif page == "About":
    st.header("ℹ️ About")
    
    st.markdown("""
    ### INBLOOM '25 Dashboard
    
    This dashboard provides comprehensive insights into the INBLOOM '25 Inter-College Cultural Events, 
    featuring participation analysis, feedback processing, and event highlights.
    
    #### Features
    - 📊 Interactive Overview Dashboard
    - 📈 Detailed Participation Analysis
    - 💬 Feedback Analysis with Word Clouds
    - 📸 Image Gallery with Processing Capabilities
    - 📋 Complete Dataset View
    
    #### Technologies Used
    - Streamlit for the web interface
    - Pandas for data manipulation
    - Matplotlib and Seaborn for visualizations
    - NLTK and spaCy for text processing
    - WordCloud for feedback visualization
    - PIL for image processing
    
    #### Created By
    **Bavirisetty Sairam**
    
    ---
    
    Built with ❤️ for INBLOOM '25
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for INBLOOM '25")
