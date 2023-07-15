import os
CLIENT_ID_TEST = 386902


def test_check_client_in_database_status_code_ok(client):
    response = client.get('/in_database', params={'client_id': CLIENT_ID_TEST})
    assert response.status_code == 200
    assert 'check' in response.json()


def test_check_client_in_database_should_return_true(client):
    response = client.get('/in_database', params={'client_id': CLIENT_ID_TEST})
    assert response.json() == {"check": True}


def test_check_client_in_database_should_return_false(client):
    response = client.get('/in_database', params={'client_id': 0})
    assert response.json() == {"check": False}


def test_get_default_threshold_should_return_numeric(client):
    response = client.get('/threshold')
    assert response.status_code == 200
    assert 'threshold' in response.json()
    assert isinstance(response.json()['threshold'], (int, float))


def test_get_client_info_status_code_ok(client):
    response = client.get('/client_info', params={"client_id": CLIENT_ID_TEST})
    assert response.status_code == 200


def test_download_client_database_status_code_ok(client):
    response = client.get('/download_database')
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/pickle'


def test_download_client_database_should_return_message(client):
    wrong_file_name = '/X_test.pkl'
    response = client.get('/download_database', params={'file_name': wrong_file_name})
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'
    assert "message" in response.json()
    assert response.json()['message'] == os.environ.get('DATA_PATH') + wrong_file_name + " file not found"


def test_predict_default_one_status_code_ok(client):
    response = client.get('/predict_default_one', params={"client_id": CLIENT_ID_TEST})
    assert response.status_code == 200


def test_predict_default_one_should_contain_keys(client):
    response = client.get('/predict_default_one', params={"client_id": CLIENT_ID_TEST})
    assert 'prediction' in response.json()
    assert 'proba_default' in response.json()


def test_predict_default_many_should_return_list(client):
    response = client.get('/predict_default_many', params={"client_ids": [CLIENT_ID_TEST]})
    assert response.status_code == 200
    assert 'proba_default' in response.json()
    assert isinstance(response.json()['proba_default'], list)


def test_get_numeric_features_should_return_list(client):
    response = client.get('/numeric_features_list')
    assert response.status_code == 200
    assert 'numeric_features' in response.json()
    assert isinstance(response.json()['numeric_features'], list)


def test_get_shap_values_should_return_three_keys(client):
    response = client.get('/shap_values_default', params={'client_id': CLIENT_ID_TEST})
    assert response.status_code == 200
    assert "shap_values" in response.json()
    assert "expected_values" in response.json()
    assert "features" in response.json()
    assert isinstance(response.json()['features'], list)


def test_get_client_ids_should_return_list_of_integers(client):
    response = client.get('/client_ids')
    assert response.status_code == 200
    assert "ids" in response.json()
    assert isinstance(response.json()["ids"], list)
    assert all(isinstance(client_id, (int, float)) for client_id in response.json()['ids'])


def test_get_nearest_neighbors_ids_should_return_list_of_n_integers(client):
    response = client.get('/nearest_neighbors_ids', params={'client_id': CLIENT_ID_TEST, 'n_neighbors': 10})
    assert response.status_code == 200
    assert "nearest_neighbors_ids" in response.json()
    assert isinstance(response.json()["nearest_neighbors_ids"], list)
    assert all(isinstance(client_id, int) for client_id in response.json()["nearest_neighbors_ids"])
