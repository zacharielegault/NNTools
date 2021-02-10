import os

from mlflow.tracking.client import MlflowClient

from nntools.utils.io import create_folder


class Tracker:
    def __init__(self, exp_name, run_id=None):
        self.exp_name = exp_name
        self.run_id = run_id
        self.save_paths = {}
        self.current_iteration = 0

    def add_path(self, key, path):
        self.save_paths[key] = path
        create_folder(path)
        self.__dict__.update(self.save_paths)

    def set_run_folder(self, path):
        self.add_path('run_folder', path)

    def create_client(self, path):
        self.client = MlflowClient(path)

        exp = self.client.get_experiment_by_name(self.exp_name)
        if exp is None:
            self.exp_id = self.client.create_experiment(self.exp_name)
        else:
            self.exp_id = exp.experiment_id

    def create_run(self, tags=None):
        if tags is None:
            tags = {}
        run = self.client.create_run(experiment_id=self.exp_id, tags=tags)
        self.run_id = run.info.run_id
        create_folder(self.run_folder)

    def get_run(self, id=None):
        if id is not None:
            self.run_id = id
        return self.client.get_run(self.run_id)

    def set_status(self, status):
        self.client.set_terminated(self.run_id, status)

    def go_to_exp_last_iteration(self):
        run = self.get_run()
        for k, v in run.data.metrics.items():
            his = self.client.get_metric_history(self.run_id, k)
            self.current_iteration = max(self.current_iteration, his[-1].step)

    def init_default_path(self):
        assert 'run_folder' in self.save_paths

        self.add_path('network_savepoint', os.path.join(self.run_folder, 'trained_model', str(self.run_id)))
        self.add_path('prediction_savepoint', os.path.join(self.run_folder, 'predictions', str(self.run_id)))

        create_folder(self.network_savepoint)
        create_folder(self.prediction_savepoint)