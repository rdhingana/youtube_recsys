"""
Embedding Generation DAG

Generates embeddings for new videos and users.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


# Default arguments
default_args = {
    'owner': 'recsys',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

# DAG definition
dag = DAG(
    'embedding_generation',
    default_args=default_args,
    description='Generate embeddings for videos and users',
    schedule_interval='0 4 * * *',  # Run at 4 AM daily (after data refresh)
    start_date=days_ago(1),
    catchup=False,
    tags=['embeddings', 'ml'],
)


# ============================================
# Task Functions
# ============================================

def check_pending_embeddings(**context):
    """Check how many items need embeddings."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    from scripts.generate_embeddings import get_stats
    
    stats = get_stats()
    
    videos_pending = stats['total_videos'] - stats['videos_with_embeddings']
    users_pending = stats['total_users'] - stats['users_with_embeddings']
    
    print(f"Videos pending embeddings: {videos_pending}")
    print(f"Users pending embeddings: {users_pending}")
    
    context['ti'].xcom_push(key='videos_pending', value=videos_pending)
    context['ti'].xcom_push(key='users_pending', value=users_pending)
    
    return {'videos': videos_pending, 'users': users_pending}


def generate_video_embeddings(**context):
    """Generate embeddings for videos without them."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    videos_pending = context['ti'].xcom_pull(key='videos_pending', task_ids='check_pending')
    
    if videos_pending == 0:
        print("No videos need embeddings")
        return 0
    
    from scripts.generate_embeddings import generate_video_embeddings as gen_video_emb
    
    # Generate embeddings (skip thumbnails for speed)
    gen_video_emb(limit=100, skip_thumbnails=True)
    
    return videos_pending


def generate_user_embeddings(**context):
    """Generate embeddings for users without them."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    users_pending = context['ti'].xcom_pull(key='users_pending', task_ids='check_pending')
    
    if users_pending == 0:
        print("No users need embeddings")
        return 0
    
    from scripts.generate_embeddings import generate_user_embeddings as gen_user_emb
    
    gen_user_emb(limit=500)
    
    return users_pending


def rebuild_faiss_index(**context):
    """Rebuild FAISS index with new embeddings."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    from scripts.build_index import load_video_embeddings
    from models.retrieval import SimpleRetrievalService, FAISSIndex
    from pathlib import Path
    
    # Load embeddings
    video_ids, video_embeddings = load_video_embeddings()
    
    if len(video_ids) == 0:
        print("No video embeddings found")
        return False
    
    print(f"Building index with {len(video_ids)} videos...")
    
    # Create service and build index
    embedding_dim = video_embeddings.shape[1]
    service = SimpleRetrievalService(embedding_dim=embedding_dim)
    
    service.index = FAISSIndex(
        embedding_dim=embedding_dim,
        index_type="IVF",
        nlist=min(100, len(video_ids) // 10),
        nprobe=10,
    )
    service.build_index(video_ids, video_embeddings)
    
    # Save index
    output_path = Path(project_root) / 'models/retrieval/saved/simple_faiss_index'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    service.save(str(output_path))
    
    print(f"Index saved to {output_path}")
    return True


def log_embedding_stats(**context):
    """Log final embedding statistics."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    from scripts.generate_embeddings import get_stats
    
    stats = get_stats()
    
    print("="*50)
    print("Embedding Generation Complete")
    print("="*50)
    print(f"Videos with embeddings: {stats['videos_with_embeddings']}/{stats['total_videos']}")
    print(f"Users with embeddings: {stats['users_with_embeddings']}/{stats['total_users']}")
    print("="*50)
    
    return stats


# ============================================
# Tasks
# ============================================

check_task = PythonOperator(
    task_id='check_pending',
    python_callable=check_pending_embeddings,
    dag=dag,
)

video_emb_task = PythonOperator(
    task_id='generate_video_embeddings',
    python_callable=generate_video_embeddings,
    dag=dag,
)

user_emb_task = PythonOperator(
    task_id='generate_user_embeddings',
    python_callable=generate_user_embeddings,
    dag=dag,
)

index_task = PythonOperator(
    task_id='rebuild_faiss_index',
    python_callable=rebuild_faiss_index,
    dag=dag,
)

stats_task = PythonOperator(
    task_id='log_stats',
    python_callable=log_embedding_stats,
    dag=dag,
)

# Task dependencies
check_task >> video_emb_task >> user_emb_task >> index_task >> stats_task