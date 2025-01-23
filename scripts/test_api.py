import unittest
import os
import json
from api import app
from model import model_train, model_predict, model_load
from logger import update_predict_log, update_train_log

class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_1_train(self):
        response = self.app.post('/train', json={"data_dir": "../data/cs-train", "test": True})
        self.assertEqual(response.status_code, 200)
        self.assertIn("training started", str(response.data))

    def test_2_predict(self):
        response = self.app.post('/predict', json={"country": "all", "year": 2019, "month": 7, "day": 1, "test": True})
        self.assertEqual(response.status_code, 200)
        self.assertIn("y_pred", str(response.data))

    def test_3_logs_train(self):
        response = self.app.get('/logs?type=train&test=True')
        self.assertEqual(response.status_code, 200)
        self.assertIn("logs", str(response.data))

    def test_4_logs_predict(self):
        response = self.app.get('/logs?type=predict&test=True')
        self.assertEqual(response.status_code, 200)
        self.assertIn("logs", str(response.data))

class ModelTestCase(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("..", "data", "cs-train")
        self.test_data = {
            "country": "all",
            "year": 2019,
            "month": 7,
            "day": 1,
            "test": True
        }

    def test_model_train_rf(self):
        result = model_train(self.data_dir, model_type='rf', test=True)
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(os.path.join("models", "test-all-0_1.joblib")))

    def test_model_predict_rf(self):
        model_train(self.data_dir, model_type='rf', test=True)
        result = model_predict(**self.test_data)
        self.assertIsNotNone(result)
        self.assertIn("y_pred", result)

    def test_model_load_rf(self):
        model_train(self.data_dir, model_type='rf', test=True)
        all_data, all_models = model_load(test=True)
        self.assertIsNotNone(all_data)
        self.assertIsNotNone(all_models)
        self.assertIn("all", all_models)

    def test_model_train_lstm(self):
        result = model_train(self.data_dir, model_type='lstm', test=True)
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(os.path.join("models", "test-all-0_1.joblib")))

    def test_model_predict_lstm(self):
        model_train(self.data_dir, model_type='lstm', test=True)
        result = model_predict(**self.test_data)
        self.assertIsNotNone(result)
        self.assertIn("y_pred", result)

    def test_model_load_lstm(self):
        model_train(self.data_dir, model_type='lstm', test=True)
        all_data, all_models = model_load(test=True)
        self.assertIsNotNone(all_data)
        self.assertIsNotNone(all_models)
        self.assertIn("all", all_models)


class LoggingTestCase(unittest.TestCase):

    def test_logging_train_read(self):
        with open("logs/train.log", "r") as file:
            logs = file.read()
            self.assertIn("supervised learing model for time-series", logs)

    def test_logging_predict_read(self):
        with open("logs/predict.log", "r") as file:
            logs = file.read()
            self.assertIn("0.1", logs)

    def test_logging_train_write(self):
        update_train_log("test_country", ("2022-01-01", "2022-12-31"), {'rmse': 100}, "00:01:00", 0.2, "test model", test=True)
        with open("logs/train-test.log", "r") as file:
            logs = file.read()
            self.assertIn("test_country", logs)
            self.assertIn("2022-01-01", logs)
            self.assertIn("2022-12-31", logs)
            self.assertIn("100", logs)
            self.assertIn("00:01:00", logs)
            self.assertIn("0.2", logs)
            self.assertIn("test model", logs)

    def test_logging_predict_write(self):
        update_predict_log("test_country", [12345], None, "2022-01-01", 10.5, 0.2, test=True)
        with open("logs/predict-test.log", "r") as file:
            logs = file.read()
            self.assertIn("test_country", logs)
            self.assertIn("12345", logs)
            self.assertIn("2022-01-01", logs)
            self.assertIn("10.5", logs)
            self.assertIn("0.2", logs)

if __name__ == '__main__':
   # unittest.main(defaultTest='LoggingTestCase')
   # unittest.main(defaultTest='ModelTestCase')
   # unittest.main(defaultTest='ApiTestCase')
   unittest.main()