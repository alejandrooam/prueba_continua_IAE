#from dagster import schedule

#@schedule(
#    cron_schedule="0 2 * * *",  # Todos los días a las 2 AM
#    job_name="full_training_pipeline",
#    execution_timezone="Europe/Berlin"
#)
#def nightly_retraining_schedule(context):
#    """
#    Programa para reentrenar automáticamente el modelo cada noche
#    """
#    return {}