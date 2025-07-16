import mlflow

def init_mlflow(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

class MLFlowLogger:
    def __init__(self, tracking_uri, experiment_name, run_name=None, params=None):
        init_mlflow(tracking_uri, experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        if params is not None:
            self.log_params(params)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    def log_loss(self, loss, step=None):
        mlflow.log_metric("loss", loss, step=step)

    def end(self):
        mlflow.end_run()


if __name__ == "__main__":
    logger = MLFlowLogger(tracking_uri="http://localhost:6100", experiment_name="test_log")
    logger.log_params({"lr": 0.00001})
    logger.log_loss(0.123, step=1)
    for epoch in range(10):
        ncut = 1.5 - epoch*epoch/100
        balance = 0.1 - epoch/100
        loss = ncut + balance
        logger.log_metrics({"ncut": ncut, "balance": balance}, step=epoch)
        logger.log_loss(loss, step=epoch)
    
    logger.end() 