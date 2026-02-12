"""
Model Retraining DAG

Retrains the two-tower retrieval model weekly.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago


# Default arguments
default_args = {
    'owner': 'recsys',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=15),
}

# DAG definition
dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Weekly model retraining pipeline',
    schedule_interval='0 6 * * 0',  # Run at 6 AM every Sunday
    start_date=days_ago(7),
    catchup=False,
    tags=['recsys','ml', 'training', 'weekly'],
)


# ============================================
# Configuration
# ============================================

MIN_INTERACTIONS_FOR_TRAINING = 1000
TRAINING_EPOCHS = 10
BATCH_SIZE = 256


# ============================================
# Task Functions
# ============================================

def check_data_readiness(**context):
    """Check if we have enough data for training."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    from scripts.load_data import get_stats
    from scripts.generate_embeddings import get_stats as get_emb_stats
    
    data_stats = get_stats()
    emb_stats = get_emb_stats()
    
    print(f"Total interactions: {data_stats['interactions']}")
    print(f"Videos with embeddings: {emb_stats['videos_with_embeddings']}")
    print(f"Users with embeddings: {emb_stats['users_with_embeddings']}")
    
    # Check if we have enough data
    has_enough_data = (
        data_stats['interactions'] >= MIN_INTERACTIONS_FOR_TRAINING and
        emb_stats['videos_with_embeddings'] >= 50 and
        emb_stats['users_with_embeddings'] >= 10
    )
    
    context['ti'].xcom_push(key='data_stats', value=data_stats)
    context['ti'].xcom_push(key='emb_stats', value=emb_stats)
    
    if has_enough_data:
        return 'train_model'
    else:
        print("Not enough data for training")
        return 'skip_training'


def train_two_tower_model(**context):
    """Train the two-tower retrieval model."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    import torch
    from torch.utils.data import DataLoader
    
    from scripts.train_retrieval import (
        load_embeddings_from_db,
        load_interactions_from_db,
        InteractionDataset,
        train_model,
    )
    from models.retrieval.two_tower import TwoTowerModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # Load data
    user_embeddings, video_embeddings = load_embeddings_from_db()
    interactions = load_interactions_from_db()
    
    if len(interactions) < MIN_INTERACTIONS_FOR_TRAINING:
        print(f"Only {len(interactions)} interactions, skipping training")
        return False
    
    # Split data
    import numpy as np
    np.random.seed(42)
    np.random.shuffle(interactions)
    
    split_idx = int(len(interactions) * 0.9)
    train_interactions = interactions[:split_idx]
    val_interactions = interactions[split_idx:]
    
    # Create datasets
    train_dataset = InteractionDataset(user_embeddings, video_embeddings, train_interactions)
    val_dataset = InteractionDataset(user_embeddings, video_embeddings, val_interactions)
    
    if len(train_dataset) == 0:
        print("No valid training data")
        return False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=min(BATCH_SIZE, len(train_dataset)), 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=min(BATCH_SIZE, len(val_dataset)), 
        shuffle=False, 
        drop_last=True
    ) if len(val_dataset) > 0 else None
    
    # Create and train model
    model = TwoTowerModel(
        user_input_dim=256,
        video_input_dim=256,
        embedding_dim=128,
    )
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=TRAINING_EPOCHS,
        learning_rate=1e-3,
        device=device,
    )
    
    # Save model
    from pathlib import Path
    output_dir = Path(project_root) / 'models/retrieval/saved'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'two_tower_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    context['ti'].xcom_push(key='model_trained', value=True)
    return True


def evaluate_model(**context):
    """Evaluate the trained model."""
    import sys
    import os
    
    project_root = os.path.dirname(os.environ.get('AIRFLOW_HOME', os.getcwd()))
    sys.path.insert(0, project_root)
    
    import torch
    import numpy as np
    
    from scripts.train_retrieval import (
        load_embeddings_from_db,
        load_interactions_from_db,
        evaluate_recall,
    )
    from models.retrieval.two_tower import TwoTowerModel
    from pathlib import Path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model_path = Path(project_root) / 'models/retrieval/saved/two_tower_model.pt'
    if not model_path.exists():
        print("No trained model found")
        return None
    
    model = TwoTowerModel(
        user_input_dim=256,
        video_input_dim=256,
        embedding_dim=128,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load data for evaluation
    user_embeddings, video_embeddings = load_embeddings_from_db()
    interactions = load_interactions_from_db()
    
    # Use last 10% as test set
    np.random.seed(42)
    np.random.shuffle(interactions)
    test_interactions = interactions[int(len(interactions) * 0.9):]
    
    # Evaluate
    recalls = evaluate_recall(
        model=model,
        user_embeddings=user_embeddings,
        video_embeddings=video_embeddings,
        test_interactions=test_interactions,
        k_values=[10, 50, 100],
        device=device,
    )
    
    # Log results
    results = {k: float(np.mean(v)) for k, v in recalls.items()}
    print(f"Evaluation results: {results}")
    
    context['ti'].xcom_push(key='eval_results', value=results)
    return results


def log_training_complete(**context):
    """Log training completion status."""
    model_trained = context['ti'].xcom_pull(key='model_trained', task_ids='train_model')
    eval_results = context['ti'].xcom_pull(key='eval_results', task_ids='evaluate_model')
    
    print("="*50)
    print("Model Retraining Complete")
    print("="*50)
    print(f"Model trained: {model_trained}")
    if eval_results:
        print("Evaluation Results:")
        for k, v in eval_results.items():
            print(f"  Recall@{k}: {v:.4f}")
    print("="*50)


# ============================================
# Tasks
# ============================================

check_task = BranchPythonOperator(
    task_id='check_data_readiness',
    python_callable=check_data_readiness,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_two_tower_model,
    dag=dag,
)

skip_task = EmptyOperator(
    task_id='skip_training',
    dag=dag,
)

eval_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

log_task = PythonOperator(
    task_id='log_complete',
    python_callable=log_training_complete,
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# Task dependencies
check_task >> [train_task, skip_task]
train_task >> eval_task >> log_task
skip_task >> log_task