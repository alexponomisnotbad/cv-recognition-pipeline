"""Airflow DAG для обучения ResNet классификатора.

Этот DAG выполняет полный цикл:
1. Проверка наличия нового датасета
2. Обучение ResNet с MLflow трекингом
3. Экспорт в ONNX
4. Валидация ONNX модели
5. (опционально) Деплой в продакшн

Запуск:
    - По расписанию (cron)
    - Вручную через Airflow UI
    - Через API: airflow dags trigger resnet_training

Конфигурация через Airflow Variables:
    - resnet_dataset_root: путь к датасету
    - resnet_onnx_output: путь для ONNX модели
    - mlflow_tracking_uri: URI MLflow сервера
"""

from datetime import datetime, timedelta
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule


# Default args для всех tasks
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def train_resnet_task(**context) -> Dict[str, Any]:
    """Task для обучения ResNet."""
    import sys
    sys.path.insert(0, "/app")

    from train_resnet_mlflow import train_and_export_onnx

    # Получаем параметры из Airflow Variables или dag_run.conf
    conf = context.get("dag_run").conf or {}

    dataset_root = conf.get(
        "dataset_root",
        Variable.get("resnet_dataset_root", default_var="/app/classification")
    )
    onnx_out = conf.get(
        "onnx_out",
        Variable.get("resnet_onnx_output", default_var="/app/models/onnx/resnet_classifier.onnx")
    )
    mlflow_uri = conf.get(
        "mlflow_tracking_uri",
        Variable.get("mlflow_tracking_uri", default_var="http://mlflow:5000")
    )
    epochs = int(conf.get("epochs", Variable.get("resnet_epochs", default_var="60")))
    batch_size = int(conf.get("batch_size", Variable.get("resnet_batch_size", default_var="4")))
    lr = float(conf.get("lr", Variable.get("resnet_lr", default_var="1e-5")))

    result = train_and_export_onnx(
        dataset_root=dataset_root,
        onnx_out=onnx_out,
        mlflow_tracking_uri=mlflow_uri,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        register_model=True,
        model_name="resnet-classifier",
    )

    # Push результат для downstream tasks
    context["ti"].xcom_push(key="training_result", value=result)

    return result


def validate_onnx_task(**context) -> bool:
    """Task для валидации ONNX модели."""
    import onnx
    import onnxruntime as ort
    import numpy as np

    ti = context["ti"]
    result = ti.xcom_pull(task_ids="train_resnet", key="training_result")
    onnx_path = result["onnx_path"]

    # Проверка структуры ONNX
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # Проверка inference
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output = session.run(None, {input_name: dummy_input})

    assert output[0].shape == (1, 3), f"Unexpected output shape: {output[0].shape}"

    print(f"ONNX validation passed: {onnx_path}")
    return True


def notify_completion_task(**context) -> None:
    """Task для уведомления о завершении."""
    ti = context["ti"]
    result = ti.xcom_pull(task_ids="train_resnet", key="training_result")

    message = f"""
    ResNet Training Completed!

    MLflow Run ID: {result['run_id']}
    ONNX Path: {result['onnx_path']}
    Best Val Accuracy: {result['best_val_acc']:.4f}
    Model Version: {result.get('model_version', 'N/A')}
    """
    print(message)

    # Здесь можно добавить отправку в Slack/Teams/Email


# DAG Definition
with DAG(
    dag_id="resnet_training",
    default_args=default_args,
    description="Train ResNet classifier with MLflow tracking",
    schedule_interval=None,  # Manual trigger или "@weekly"
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "resnet", "mlflow"],
    params={
        "dataset_root": "/app/classification",
        "epochs": 60,
        "batch_size": 4,
        "lr": 1e-5,
    },
) as dag:

    # Task 1: Проверка датасета
    check_dataset = BashOperator(
        task_id="check_dataset",
        bash_command="""
            DATASET_ROOT="{{ params.dataset_root }}"
            if [ ! -f "$DATASET_ROOT/annotations.xml" ]; then
                echo "Error: annotations.xml not found in $DATASET_ROOT"
                exit 1
            fi
            if [ ! -d "$DATASET_ROOT/images" ]; then
                echo "Error: images directory not found in $DATASET_ROOT"
                exit 1
            fi
            echo "Dataset OK: $(ls $DATASET_ROOT/images | wc -l) images found"
        """,
    )

    # Task 2: Обучение ResNet
    train_resnet = PythonOperator(
        task_id="train_resnet",
        python_callable=train_resnet_task,
        provide_context=True,
    )

    # Task 3: Валидация ONNX
    validate_onnx = PythonOperator(
        task_id="validate_onnx",
        python_callable=validate_onnx_task,
        provide_context=True,
    )

    # Task 4: Уведомление
    notify_completion = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_completion_task,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Dependencies
    check_dataset >> train_resnet >> validate_onnx >> notify_completion
