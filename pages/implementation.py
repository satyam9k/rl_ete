import streamlit as st
import numpy as np
import random
import plotly.express as px
import pandas as pd

def implementation_page():
    st.title("🗓️ Personal Time Management")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Time Input
    start_time = st.sidebar.slider(
        "Start of Free Time", 
        min_value=0, 
        max_value=23, 
        value=9,
        help="When do you start your productive hours?"
    )
    
    end_time = st.sidebar.slider(
        "End of Free Time", 
        min_value=0, 
        max_value=23, 
        value=17,
        help="When do your productive hours end?"
    )
    
    hours_free = end_time - start_time
    
    # Task Input
    num_tasks = st.sidebar.number_input(
        "Number of Tasks", 
        min_value=1, 
        max_value=10, 
        value=3
    )
    
    tasks = []
    for i in range(num_tasks):
        with st.sidebar.expander(f"Task {i+1}"):
            name = st.text_input(f"Task {i+1} Name")
            duration = st.number_input(
                f"Duration (hours)", 
                min_value=0.5, 
                max_value=float(hours_free), 
                value=1.0,
                step=0.5
            )
            priority = st.slider(
                f"{name} Priority", 
                min_value=1, 
                max_value=10, 
                value=5
            )
            
            if name:
                tasks.append({
                    "name": name, 
                    "priority": priority, 
                    "duration": int(duration),
                    "due_in": hours_free
                })
    
    # Reward Function
    def reward_function(task, action, time_left):
        if not tasks:
            return 0
        if action == len(tasks):  # Break action
            return 1  # Encourage taking breaks
        if time_left < task["duration"]:
            return -10  # Task can't fit in the remaining time
        urgency = max(1, (task["priority"] * (10 - task["due_in"])) / 10)
        return urgency
    
    # Train Model Function
    def train_model(hours_free, tasks):
        time_slots = hours_free
        num_tasks = len(tasks)
        actions = range(num_tasks + 1)
        Q_table = np.zeros((time_slots, len(actions)))
        gamma, alpha, epsilon = 0.9, 0.1, 0.2

        for episode in range(1000):
            state = time_slots - 1
            while state >= 0:
                action = np.argmax(Q_table[state]) if random.random() > epsilon else random.choice(actions)

                if action < len(tasks):
                    task = tasks[action]
                    reward = reward_function(task, action, state)
                    state_prime = max(0, state - task["duration"])
                else:
                    reward = reward_function(None, action, state)
                    state_prime = state - 1

                Q_table[state, action] += alpha * (
                    reward + gamma * np.max(Q_table[state_prime]) - Q_table[state, action]
                )
                state = state_prime

        return Q_table
    
    # Recommend Tasks
    def recommend_tasks(Q_table, tasks, hours_free):
        current_time = hours_free - 1
        schedule = []
        while current_time >= 0:
            action = np.argmax(Q_table[current_time])
            if action == len(tasks):
                schedule.append("Break")
                current_time -= 1
            else:
                task = tasks[action]
                if current_time >= task["duration"]:
                    schedule.append(task["name"])
                    current_time -= task["duration"]
                else:
                    break
        return schedule
    
    # Return to About Page
    if st.sidebar.button("About this Tool"):
        st.switch_page("main.py")
    
    # Generate Schedule Button
    if st.sidebar.button("Generate Optimal Schedule"):
        if tasks:
            Q_table = train_model(hours_free, tasks)
            schedule = recommend_tasks(Q_table, tasks, hours_free)
            
            st.header("📋 Your Optimized Schedule")
            schedule_df = pd.DataFrame({
                'Hour': [f"{start_time + i}:00" for i in range(len(schedule))],
                'Activity': schedule
            })
            
            st.dataframe(schedule_df, use_container_width=True)
            
            # Pie chart of time allocation
            activity_counts = schedule_df['Activity'].value_counts()
            fig = px.pie(
                activity_counts, 
                values=activity_counts.values, 
                names=activity_counts.index,
                title='Time Allocation'
            )
            st.plotly_chart(fig)
        else:
            st.error("Please add some tasks first!")

def main():
    implementation_page()

if __name__ == "__main__":
    main()
