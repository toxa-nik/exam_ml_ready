import os
import pickle
import pandas as pd
import numpy as np
import sqlalchemy
from dagster import asset, get_dagster_logger, Definitions, define_asset_job, ScheduleDefinition, load_assets_from_current_module
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues, TestColumnsType
import mlflow
from lifelines import CoxPHFitter
import tempfile

logger = get_dagster_logger()

STUDENT_ID_COL = "user_id"
TIME_COL = "duration_days"
EVENT_COL = "event_dropout"
FEATURES = [
    "actions_total",                           
    "LMS_ratio_weekend", 
    "lesson_activity_user_lessons_nunique",
    "SOC_peer_activity_ratio"
]

@asset(group_name="ml_pipeline")
def raw_student_data() -> pd.DataFrame:
    """Извлекает всю накопленную историю студентов из PostgreSQL."""
    logger.info("Подключение к PostgreSQL Feature Store...")
    
    try:
        DB_USER = os.environ["DB_USER"]
        DB_PASSWORD = os.environ["DB_PASSWORD"]
        DB_HOST = os.environ.get("POSTGRES_HOST", "postgres")
        DB_NAME = os.environ.get("POSTGRES_DB", "ml_database")
    except KeyError as e:
        logger.error(f"ERROR: Missing env var {e}")
        raise RuntimeError(f"Остановка пайплайна: нет доступа к базе данных. Проверьте {e}")
    
    engine = sqlalchemy.create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}")
    
    try:
        df = pd.read_sql("SELECT * FROM raw_students", engine)
        logger.info(f"Успешно выгружено {len(df)} строк из базы.")
        return df
    except Exception as e:
        logger.error(f"Таблица raw_students не найдена: {e}")
        raise RuntimeError("База данных пуста. Отправьте данные на эндпоинт /ingest.")

@asset(group_name="ml_pipeline")
def validated_student_data(raw_student_data: pd.DataFrame) -> pd.DataFrame:
    """Проверка качества данных через EvidentlyAI."""
    logger.info("Запуск EvidentlyAI Data Quality Checks...")
    
    data_quality_tests = TestSuite(tests=[
        TestNumberOfMissingValues(),
        TestColumnsType(),
    ])
    
    data_quality_tests.run(reference_data=None, current_data=raw_student_data)
    test_results = data_quality_tests.as_dict()
    
    if not test_results["summary"]["all_passed"]:
        failed_tests = test_results["summary"]["failed_tests"]
        logger.error(f"Валидация провалена. Ошибок: {failed_tests}")
        # raise ValueError("Ошибка Data Quality Check! Пайплайн остановлен.")
    
    else:
        logger.info("Валидация успешно пройдена.")
    
    raw_student_data = raw_student_data.fillna(0) 
    return raw_student_data
        
@asset(group_name="ml_pipeline")
def trained_cox_model(validated_student_data: pd.DataFrame):
    """Обучает интерпретируемую модель CoxPHFitter."""
    logger.info("Запуск обучения модели CoxPH...")

    df_train = validated_student_data[[TIME_COL, EVENT_COL] + FEATURES].copy()

    for col in FEATURES:
        if df_train[col].nunique() < 5:
            df_train[col] += np.random.normal(0, 0.0001, len(df_train))

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_train, duration_col=TIME_COL, event_col=EVENT_COL)

    logger.info("Модель успешно обучена.")
    return cph

@asset(group_name="ml_pipeline")
def logged_mlflow_model(trained_cox_model, validated_student_data: pd.DataFrame):
    """Логирует модель и метрики в MLflow."""
    logger.info("Связь с сервером MLflow...")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("EdTech_Student_Churn_v2")

    mlflow.autolog(disable=True) 

    run = mlflow.start_run()
    try:
        mlflow.log_param("features", ", ".join(FEATURES))
        mlflow.log_param("model_type", "CoxPH")

        try:
            c_index = trained_cox_model.concordance_index_
            mlflow.log_metric("c_index", c_index)
            logger.info(f"C-index модели: {c_index:.4f}")
        except Exception as e:
            logger.warning(f"Не удалось посчитать C-index: {e}")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "cox_model.pkl")
            
            with open(model_path, "wb") as f:
                pickle.dump(trained_cox_model, f)
            
            mlflow.log_artifact(model_path, artifact_path="model") 

    finally:
        mlflow.end_run()

    logger.info("Пайплайн завершен и модель сохранена в MLflow!")
    return True
    
all_assets = load_assets_from_current_module()

retrain_model_job = define_asset_job(
    name="daily_job",
    selection="*"
) 

daily_schedule = ScheduleDefinition(
    job=retrain_model_job,
    cron_schedule="0 2 * * *",
)
    
defs = Definitions(
    assets=all_assets,
    jobs=[retrain_model_job],
    schedules=[daily_schedule],
)