import pytest
from unittest.mock import patch, MagicMock
from main import main

@patch("main.load_data")
@patch("main.preprocess_data")
@patch("main.train_model")
@patch("main.evaluate")
@patch("main.joblib.dump")
@patch("main.mlflow.start_run")
@patch("main.mlflow.set_tracking_uri")
@patch("main.mlflow.set_experiment")
def test_main_pipeline(
    mock_set_experiment,
    mock_set_uri,
    mock_start_run,
    mock_dump,
    mock_evaluate,
    mock_train_model,
    mock_preprocess_data,
    mock_load_data
):
    # Mocks
    mock_load_data.return_value = ("train_df", "test_df")
    mock_preprocess_data.return_value = (
        [[0.1]*5]*2, [[0.1]*5]*2, [0, 1], [0, 1], MagicMock(), {"encoder": MagicMock()}
    )
    mock_train_model.return_value = MagicMock()
    mock_start_run.return_value.__enter__.return_value = MagicMock()

    # Run main
    main()

    # Assertions
    mock_load_data.assert_called_once()
    mock_preprocess_data.assert_called_once()
    mock_train_model.assert_called_once()
    mock_evaluate.assert_called_once()
    mock_dump.assert_called()
