import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page config for better look
st.set_page_config(page_title="DRDO RAC Expert Matcher", layout="wide")

# Title and Description
st.title("🛡️ DRDO RAC Expert Matching System")
st.markdown("""
This prototype automates matching subject experts to interview boards based on candidate expertise and interview subject.  
**Inputs**: Candidate keywords (e.g., 'AI, Machine Learning') and Interview Subject (e.g., 'Aerospace Engineering').  
**Output**: Top 5 recommended experts with match scores (0-1; higher is better).  
*Built for Project-Based Learning – Original work by [Samruddhi kadam and team].*
""")

# Load data (Backend: Data Layer)
@st.cache_data  # Cache for speed
def load_experts():
    try:
        experts = pd.read_csv('experts.csv')
        return experts
    except FileNotFoundError:
        st.error("experts.csv not found! Create it with sample data.")
        return pd.DataFrame()

experts_df = load_experts()
if not experts_df.empty:
    st.success(f"Loaded {len(experts_df)} experts from database.")
    # Show sample data sidebar
    with st.sidebar:
        st.subheader("Sample Experts")
        st.dataframe(experts_df.head())

# Frontend: Input Forms
st.header("🔍 Enter Details for Matching")
col1, col2 = st.columns(2)

with col1:
    candidate_keywords = st.text_input(
        "Candidate Expertise Keywords (comma-separated)",
        placeholder="e.g., AI, Machine Learning, Defense Tech",
        help="Extract from candidate resume."
    )

with col2:
    interview_subject = st.text_input(
        "Interview Subject",
        placeholder="e.g., Aerospace Engineering",
        help="Main topic of the interview."
    )

# Backend: Matching Function
def match_experts(candidate_keywords, interview_subject, experts_df):
    if pd.isna(candidate_keywords) or pd.isna(interview_subject) or experts_df.empty:
        return pd.DataFrame()
    
    # Combine query
    query = f"{candidate_keywords}, {interview_subject}"
    
    # Prepare texts: query + all expert keywords
    all_texts = [query] + list(experts_df['Expertise_Keywords'].fillna(''))
    
    # Vectorize (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compute similarities (Backend Core)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    # Add scores to dataframe
    experts_df = experts_df.copy()
    experts_df['Match_Score'] = similarities
    experts_df['Match_Score'] = np.round(experts_df['Match_Score'], 3)  # Round for display
    
    # Filter available experts and rank top 5
    available_experts = experts_df[experts_df['Availability'] == 'Yes']
    if available_experts.empty:
        available_experts = experts_df  # Fallback to all if none available
    
    recommendations = available_experts.nlargest(5, 'Match_Score')[['Name', 'Expertise_Keywords', 'Affiliation', 'Availability', 'Match_Score']]
    
    return recommendations

# Run Matching on Button Click
if st.button("🚀 Find Matches", type="primary"):
    if candidate_keywords and interview_subject:
        with st.spinner("Computing matches... (Using TF-IDF + Cosine Similarity)"):
            results = match_experts(candidate_keywords, interview_subject, experts_df)
        
        if not results.empty:
            st.header("📋 Recommendations")
            st.markdown("**Top Experts (Ranked by Match Score)**")
            st.dataframe(results, use_container_width=True)
            
            # Visual: Bar chart of scores
            st.subheader("Match Score Visualization")
            chart_data = results.set_index('Name')['Match_Score']
            st.bar_chart(chart_data)
            
            # Explanation
            st.info(f"""
            **How it Works**:  
            - Query: "{candidate_keywords}, {interview_subject}"  
            - Algorithm: TF-IDF vectors text into math representations, then cosine similarity finds the closest matches.  
            - Example Score: >0.7 = Strong match (e.g., shared keywords like 'Aerospace').  
            - Next: Add resume upload for auto-keyword extraction.
            """)
        else:
            st.warning("No matches found. Try broader keywords!")
    else:
        st.warning("Please fill both inputs.")

# Footer
st.markdown("---")
st.markdown("*Prototype v1.0 | For DRDO RAC Interviews | 80% Complete: Core Matching + UI Done.*")