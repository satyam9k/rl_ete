import streamlit as st
import numpy as np
import random
import plotly.express as px
import pandas as pd
from typing import List, Dict
import uuid

class SmartScheduler:
    def __init__(self):
        # Initialize session state for persistent data
        if 'task_history' not in st.session_state:
            st.session_state.task_history = []
        if 'preferences' not in st.session_state:
            st.session_state.preferences = {
                'break_frequency': 2,
                'max_task_duration': 3,
                'energy_curve': [
                    (0, 0.5),   # Morning start
                    (3, 0.8),   # Mid-morning peak
                    (6, 0.6),   # Afternoon dip
                    (9, 0.7)    # Evening recovery
                ]
            }

    def advanced_reward_function(self, task: Dict, context: Dict) -> float:
        """
        Multi-dimensional reward function considering:
        1. Task priority
        2. Energy levels
        3. Previous performance
        4. Task diversity
        """
        # Base priority reward
        priority_score = task['priority'] * 2
        
        # Energy level adjustment
        current_hour = context.get('current_hour', 0)
        energy_multiplier = self._get_energy_multiplier(current_hour)
        
        # Task history consideration
        task_diversity_bonus = self._calculate_task_diversity_bonus(task)
        
        # Recency penalty for recently completed similar tasks
        recency_penalty = self._calculate_recency_penalty(task)
        
        # Combine factors
        final_reward = (
            priority_score * energy_multiplier + 
            task_diversity_bonus - 
            recency_penalty
        )
        
        return max(0, final_reward)

    def _get_energy_multiplier(self, current_hour: int) -> float:
        """
        Calculate energy multiplier based on time of day
        """
        preferences = st.session_state.preferences['energy_curve']
        for (hour, multiplier) in preferences:
            if current_hour <= hour:
                return multiplier
        return 0.5

    def _calculate_task_diversity_bonus(self, task: Dict) -> float:
        """
        Encourage diversity in task types
        """
        task_history = st.session_state.task_history
        if not task_history:
            return 1.0
        
        recent_tasks = task_history[-3:]
        similar_tasks = [t for t in recent_tasks if t['category'] == task.get('category')]
        
        # More bonus for less similar recent tasks
        return 1 + (3 - len(similar_tasks)) * 0.2

    def _calculate_recency_penalty(self, task: Dict) -> float:
        """
        Penalize repeating similar tasks too frequently
        """
        task_history = st.session_state.task_history
        if not task_history:
            return 0
        
        recent_similar = [
            t for t in task_history[-3:] 
            if t['name'] == task['name']
        ]
        
        return len(recent_similar) * 0.5

    def advanced_q_learning(self, tasks: List[Dict], hours_free: int) -> np.ndarray:
        """
        Enhanced Q-Learning with more sophisticated exploration
        """
        num_states = hours_free
        num_actions = len(tasks) + 2  # Tasks + break + idle
        Q_table = np.zeros((num_states, num_actions))
        
        # Advanced hyperparameters
        gamma = 0.9  # Discount factor
        alpha = 0.1  # Learning rate
        epsilon_start = 0.8
        epsilon_decay = 0.95
        
        for episode in range(1000):
            # Dynamic epsilon for exploration
            epsilon = max(0.1, epsilon_start * (epsilon_decay ** episode))
            
            state = num_states - 1
            while state > 0:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, num_actions - 1)
                else:
                    action = np.argmax(Q_table[state])
                
                # Context for reward calculation
                context = {
                    'current_hour': num_states - state,
                    'total_hours': num_states
                }
                
                # Select task or action
                if action < len(tasks):
                    task = tasks[action]
                    reward = self.advanced_reward_function(task, context)
                    state_prime = max(0, state - int(task['duration']))
                elif action == len(tasks):  # Break
                    reward = 1.0  # Encourage periodic breaks
                    state_prime = state - 1
                else:  # Idle
                    reward = 0.1  # Minimal reward for idle time
                    state_prime = state - 1
                
                # Q-learning update
                Q_table[state, action] += alpha * (
                    reward + gamma * np.max(Q_table[state_prime]) - Q_table[state, action]
                )
                
                state = state_prime
        
        return Q_table

    def recommend_schedule(self, tasks: List[Dict], hours_free: int) -> List[Dict]:
        """
        Generate a more intelligent schedule
        """
        Q_table = self.advanced_q_learning(tasks, hours_free)
        
        current_state = hours_free - 1
        schedule = []
        used_tasks = set()
        
        while current_state > 0:
            action = np.argmax(Q_table[current_state])
            
            if action < len(tasks):
                task = tasks[action]
                if task['name'] not in used_tasks and current_state >= task['duration']:
                    schedule.append({
                        'task': task['name'],
                        'start_hour': hours_free - current_state,
                        'duration': task['duration'],
                        'priority': task['priority']
                    })
                    used_tasks.add(task['name'])
                    current_state -= task['duration']
                else:
                    current_state -= 1
            elif action == len(tasks):  # Break
                schedule.append({
                    'task': 'Break',
                    'start_hour': hours_free - current_state,
                    'duration': 1,
                    'priority': 0
                })
                current_state -= 1
            else:  # Idle
                current_state -= 1
        
        # Sort schedule by start hour
        return sorted(schedule, key=lambda x: x['start_hour'])

def main_app():
    st.set_page_config(layout="wide", page_title="Smart Scheduler")
    
    st.title("🚀 Intelligent Time Management System")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Advanced Settings")
    
    # Time range selection
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.number_input(
            "Start Time", 
            min_value=0, 
            max_value=23, 
            value=9, 
            key='start_time_input'
        )
    with col2:
        end_time = st.number_input(
            "End Time", 
            min_value=0, 
            max_value=23, 
            value=17, 
            key='end_time_input'
        )
    
    hours_free = end_time - start_time
    
    # Task input with improved categorization
    st.subheader("📋 Task Input")
    
    # Predefined task categories
    task_categories = [
        "Work", "Personal Development", "Health", 
        "Creativity", "Household", "Learning"
    ]
    
    tasks = []
    num_tasks = st.number_input(
        "Number of Tasks", 
        min_value=1, 
        max_value=10, 
        value=3, 
        key='num_tasks'
    )
    
    for i in range(num_tasks):
        col1, col2, col3 = st.columns(3)
        with col1:
            task_name = st.text_input(
                f"Task {i+1} Name", 
                key=f'task_name_{i}'
            )
        with col2:
            task_category = st.selectbox(
                f"Category for {task_name or 'Task'}",
                task_categories,
                key=f'task_category_{i}'
            )
        with col3:
            task_duration = st.number_input(
                "Duration (hours)", 
                min_value=0.5, 
                max_value=float(hours_free), 
                value=1.0, 
                key=f'task_duration_{i}'
            )
        
        task_priority = st.slider(
            f"Priority for {task_name or 'Task'}", 
            min_value=1, 
            max_value=10, 
            value=5, 
            key=f'task_priority_{i}'
        )
        
        if task_name:
            tasks.append({
                'id': str(uuid.uuid4()),
                'name': task_name,
                'category': task_category,
                'duration': int(task_duration),
                'priority': task_priority
            })
        
        st.divider()
    
    # Generate Schedule Button
    if st.button("Generate Intelligent Schedule", key='generate_schedule'):
        scheduler = SmartScheduler()
        
        try:
            schedule = scheduler.recommend_schedule(tasks, hours_free)
            
            # Display Schedule
            st.subheader("🗓️ Optimized Daily Schedule")
            
            # Schedule Table
            schedule_df = pd.DataFrame(schedule)
            schedule_df['Start Time'] = schedule_df['start_hour'].apply(lambda x: f"{start_time + x}:00")
            schedule_df = schedule_df[['Start Time', 'task', 'duration', 'priority']]
            schedule_df.columns = ['Time', 'Activity', 'Duration', 'Priority']
            
            st.table(schedule_df)
            
            # Visualization
            fig = px.bar(
                schedule_df, 
                x='Activity', 
                y='Duration', 
                color='Priority',
                title='Schedule Breakdown'
            )
            st.plotly_chart(fig)
        
        except Exception as e:
            st.error(f"Error generating schedule: {e}")
    
    # Advanced Insights
    if st.checkbox("Show Advanced Insights"):
        st.subheader("🧠 Scheduling Insights")
        st.write("""
        - Our AI considers multiple factors:
          * Task Priority
          * Energy Levels Throughout the Day
          * Task Diversity
          * Recent Task History
        
        - Intelligent Break Management
        - Dynamic Exploration Strategy
        """)

if __name__ == "__main__":
    main_app()
