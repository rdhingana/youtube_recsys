"""
Daily Data Refresh DAG

Scrapes new videos and simulates user interactions daily.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


# Default arguments
default_args = {
    'owner': 'recsys',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'daily_data_refresh',
    default_args=default_args,
    description='Daily scraping and data refresh pipeline',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=['data', 'daily'],
)


# ============================================
# Task Functions
# ============================================

def scrape_trending_videos(**context):
    """Scrape trending/popular videos."""
    import sys
    import os
    
    # Get project root from AIRFLOW_HOME parent
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    from data.scraper.youtube_scraper import YouTubeScraper
    
    scraper = YouTubeScraper(output_dir=os.path.join(project_root, 'data/raw'))
    
    # Search queries for fresh content
    queries = [
        "trending today",
        "new music video",
        "tech news",
        "tutorial 2024",
    ]
    
    all_videos = []
    for query in queries:
        try:
            for video in scraper.scrape_search(query, max_videos=10):
                all_videos.append(video)
        except Exception as e:
            print(f"Error scraping '{query}': {e}")
    
    if all_videos:
        filename = f"videos_{datetime.now().strftime('%Y%m%d')}.json"
        scraper.save_to_json(all_videos, filename)
        
        # Push to XCom for next task
        context['ti'].xcom_push(key='videos_file', value=os.path.join(project_root, f'data/raw/{filename}'))
        context['ti'].xcom_push(key='video_count', value=len(all_videos))
    
    return len(all_videos)


def load_videos_to_db(**context):
    """Load scraped videos to database."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    from scripts.load_data import load_videos_from_json, insert_videos
    
    # Get file from previous task
    videos_file = context['ti'].xcom_pull(key='videos_file', task_ids='scrape_videos')
    
    if not videos_file:
        print("No videos file found")
        return 0
    
    try:
        videos = load_videos_from_json(videos_file)
        count = insert_videos(videos)
        print(f"Loaded {count} videos to database")
        return count
    except Exception as e:
        print(f"Error loading videos: {e}")
        raise


def simulate_daily_interactions(**context):
    """Simulate user interactions for the day."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    from scripts.load_data import (
        get_videos_from_db, 
        insert_users, 
        insert_interactions,
        get_stats
    )
    from data.simulator.user_simulator import UserSimulator
    
    # Get current stats
    stats = get_stats()
    print(f"Current stats: {stats}")
    
    # Get videos
    videos = get_videos_from_db()
    if not videos:
        print("No videos in database")
        return 0
    
    # Simulate a few new users
    simulator = UserSimulator(videos, seed=int(datetime.now().timestamp()))
    new_users = simulator.generate_users(5)
    insert_users(new_users)
    
    # Simulate interactions for existing users (1 day)
    # This is simplified - in production you'd load existing users
    all_interactions = []
    for user in new_users:
        interactions = simulator.simulate_history(user, days=1)
        all_interactions.extend(interactions)
    
    if all_interactions:
        count = insert_interactions(all_interactions)
        print(f"Inserted {count} interactions")
        return count
    
    return 0


def log_stats(**context):
    """Log database statistics."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    from scripts.load_data import get_stats
    
    stats = get_stats()
    print("="*50)
    print("Daily Data Refresh Complete")
    print("="*50)
    print(f"Total Videos: {stats['videos']}")
    print(f"Total Users: {stats['users']}")
    print(f"Total Interactions: {stats['interactions']}")
    print("="*50)
    
    return stats


# ============================================
# Tasks
# ============================================

scrape_task = PythonOperator(
    task_id='scrape_videos',
    python_callable=scrape_trending_videos,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_videos',
    python_callable=load_videos_to_db,
    dag=dag,
)

simulate_task = PythonOperator(
    task_id='simulate_interactions',
    python_callable=simulate_daily_interactions,
    dag=dag,
)

stats_task = PythonOperator(
    task_id='log_stats',
    python_callable=log_stats,
    dag=dag,
)

# Task dependencies
scrape_task >> load_task >> simulate_task >> stats_task