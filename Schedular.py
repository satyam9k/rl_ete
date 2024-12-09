import streamlit as st
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Reward function
def reward_function(task, action, time_left):
    if action == len(tasks):  # Break action
        return 1  # Encourage taking breaks
    if time_left < task["duration"]:
        return -10  # Task can't fit in the remaining time
    urgency = max(1, (task["priority"] * (10 - task["due_in"])) / 10)
    return urgency

# Train RL model
def train_model(hours_free, tasks):
    time_slots = hours_free
    num_tasks = len(tasks)
    actions = range(num_tasks + 1)
    Q_table = np.zeros((time_slots, len(actions)))
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.2

    for episode in range(1000):
        state = time_slots - 1
        while state >= 0:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action = np.argmax(Q_table[state])

            if action < len(tasks):
                task = tasks[action]
                reward = reward_function(task, action, state)
                state_prime = state - task["duration"] if state >= task["duration"] else state
            else:
                reward = reward_function(None, action, state)
                state_prime = state - 1

            Q_table[state, action] += alpha * (
                reward + gamma * np.max(Q_table[state_prime]) - Q_table[state, action]
            )
            state = state_prime

    return Q_table

# Recommend tasks based on Q-table
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

# Visualization of Q-table
def plot_q_table(Q_table):
    plt.figure(figsize=(10, 6))
    sns.heatmap(Q_table, cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
    plt.title("Q-Table Heatmap for Task Scheduling")
    plt.xlabel("Actions (Tasks + Break)")
    plt.ylabel("Time Slots Remaining")
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return buffer

# Streamlit App
st.set_page_config(page_title="Smart Task Scheduler", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigate", ["Information", "Task Scheduler"])

# Page 1: Information
if page == "Information":
    st.title("Smart Task Scheduler")
    st.header("Why is this tool needed?")
    st.write("""
    In today's fast-paced world, managing tasks effectively is critical for productivity and mental well-being. 
    This tool helps you:
    - Optimize your schedule based on your available time and task priorities.
    - Minimize procrastination and focus better using a reinforcement learning algorithm.
    - Recalculate schedules dynamically as tasks are completed or new time constraints arise.
    """)
    st.header("How does it work?")
    st.write("""
    The application uses a **Q-learning algorithm**, a type of reinforcement learning, to:
    - Train a model based on task priorities, durations, and available time slots.
    - Provide a schedule that balances work and breaks.
    - Adjust the schedule dynamically based on user feedback.
    """)
    st.header("How to use this app?")
    st.write("""
    1. Navigate to the **Task Scheduler** page using the sidebar.
    2. Input your available time and tasks.
    3. Review the recommended schedule.
    4. Update tasks or time slots as needed to recalculate the schedule dynamically.
    """)

# Page 2: Task Scheduler
elif page == "Task Scheduler":
    st.title("Task Scheduler")
    
    # User Inputs (handling session state for persistence)
    if 'start_time' not in st.session_state:
        st.session_state.start_time = 9
        st.session_state.end_time = 17
        st.session_state.hours_free = st.session_state.end_time - st.session_state.start_time
        st.session_state.tasks = []

    # Sidebar User Input
    st.sidebar.header("Your Available Time")
    st.session_state.start_time = st.sidebar.number_input("Start Hour (24-hour format)", min_value=0, max_value=23, value=st.session_state.start_time)
    st.session_state.end_time = st.sidebar.number_input("End Hour (24-hour format)", min_value=0, max_value=23, value=st.session_state.end_time)
    st.session_state.hours_free = st.session_state.end_time - st.session_state.start_time
    
    st.sidebar.header("Your Tasks")
    num_tasks = st.sidebar.number_input("Number of Tasks", min_value=1, max_value=10, value=max(1, len(st.session_state.tasks)))

    st.session_state.tasks = []  # Reset tasks if user changes task count
    for i in range(num_tasks):
        name = st.sidebar.text_input(f"Task {i + 1} Name", value=f"Task {i + 1}" if len(st.session_state.tasks) <= i else st.session_state.tasks[i]["name"])
        duration = st.sidebar.number_input(f"Task {i + 1} Duration (hours)", min_value=1, max_value=st.session_state.hours_free, value=1 if len(st.session_state.tasks) <= i else st.session_state.tasks[i]["duration"])
        priority = st.sidebar.slider(f"Task {i + 1} Priority (1-10)", min_value=1, max_value=10, value=5 if len(st.session_state.tasks) <= i else st.session_state.tasks[i]["priority"])
        st.session_state.tasks.append({"name": name, "priority": priority, "duration": duration, "due_in": st.session_state.hours_free})

    # Button to generate initial schedule
    if st.sidebar.button("Generate Schedule"):
        Q_table = train_model(st.session_state.hours_free, st.session_state.tasks)
        schedule = recommend_tasks(Q_table, st.session_state.tasks, st.session_state.hours_free)
        
        # Display Schedule
        st.header("Recommended Schedule")
        for hour, task in enumerate(schedule):
            st.write(f"Hour {st.session_state.start_time + hour}: {task}")
        
        # Q-Table Heatmap
        st.header("Q-Table Visualization")
        st.image(plot_q_table(Q_table), use_column_width=True)

    # Button to update schedule
    if st.sidebar.button("Update Schedule"):
        st.sidebar.header("Update Options")
        update_option = st.sidebar.selectbox("Choose an update option", ["Mark Task as Done", "Adjust Available Time"])
        
        if update_option == "Mark Task as Done":
            done_task = st.sidebar.text_input("Enter the name of the completed task")
            st.session_state.tasks = [task for task in st.session_state.tasks if task["name"] != done_task]
        
        elif update_option == "Adjust Available Time":
            new_start_time = st.sidebar.number_input("New Start Hour (24-hour format)", min_value=0, max_value=23, value=st.session_state.start_time)
            new_end_time = st.sidebar.number_input("New End Hour (24-hour format)", min_value=0, max_value=23, value=st.session_state.end_time)
            st.session_state.hours_free = new_end_time - new_start_time
            st.session_state.start_time = new_start_time
            st.session_state.end_time = new_end_time
        
        # Recalculate Schedule
        Q_table = train_model(st.session_state.hours_free, st.session_state.tasks)
        schedule = recommend_tasks(Q_table, st.session_state.tasks, st.session_state.hours_free)
        st.header("Updated Schedule")
        for hour, task in enumerate(schedule):
            st.write(f"Hour {st.session_state.start_time + hour}: {task}")
        st.header("Q-Table Visualization")
        st.image(plot_q_table(Q_table), use_column_width=True)
