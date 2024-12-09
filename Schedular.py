import streamlit as st
import numpy as np
import random
import plotly.express as px
import pandas as pd

class TimeManagementApp:
    def __init__(self):
        st.set_page_config(
            page_title="Smart Time Management Tool",
            page_icon="⏰",
            layout="wide"
        )
        
    def about_page(self):
        st.title("🕰️ Smart Time Management Tool")
        
        # Sections with colorful cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Why Do You Need This?
            - Optimize your daily productivity
            - Reduce decision fatigue
            - Prioritize tasks effectively
            - Balance work and breaks
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            ### 🧠 How It Works
            - Input your available time
            - List your tasks with priorities
            - AI learns your optimal schedule
            - Continuously adapts to your needs
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            ### 🤖 Our Approach
            - Reinforcement Learning (Q-Learning)
            - Dynamic task scheduling
            - Personalized recommendation
            - Real-time adaptability
            """, unsafe_allow_html=True)
        
        # Detailed Explanation
        st.header("The Science Behind Smart Scheduling")
        
        # Explanation Columns
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.subheader("Reinforcement Learning Magic")
            st.write("""
            Our tool uses Q-Learning, a powerful machine learning technique that:
            - Learns from your task patterns
            - Assigns values to different actions
            - Continuously improves scheduling strategy
            """)
        
        with right_col:
            # Simple visualization of Q-Learning concept
            df = pd.DataFrame({
                'State': ['High Priority', 'Medium Priority', 'Low Priority'],
                'Learning Score': [0.9, 0.6, 0.3]
            })
            fig = px.bar(df, x='State', y='Learning Score', 
                         title='Q-Learning Adaptation')
            st.plotly_chart(fig)
        
        # Algorithm Details
        with st.expander("Deep Dive: Q-Learning Algorithm"):
            st.markdown("""
            ### Q-Learning Breakdown
            1. **State**: Available time slots
            2. **Actions**: Choose a task or take a break
            3. **Reward Function**:
               - Prioritize important tasks
               - Consider task duration
               - Encourage balanced schedules
            4. **Learning Process**:
               - Explore different task combinations
               - Update action values based on rewards
               - Converge to optimal scheduling strategy
            """)
        
        # Start Button
        if st.button("Start Planning Your Day! 🚀"):
            st.switch_page("pages/implementation.py")
    
    def render(self):
        self.about_page()

# Save this as main.py in the root directory
def main():
    app = TimeManagementApp()
    app.render()

if __name__ == "__main__":
    main()
