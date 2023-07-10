import pandas as pd
import numpy as np
import os
from pytest import MonkeyPatch
import pytest

mp = MonkeyPatch()
URL_API_TEST = "http://localhost:80/"


class TestDashboard:
    @classmethod
    def setup_class(cls):
        mp.setenv('URL_API', URL_API_TEST)
        from dashboard import functions
        mp.setattr(functions, 'store_request', cls.mock_action)

    @classmethod
    def teardown_class(cls):
        mp.delenv('URL_API')
        if os.path.isfile('database.pkl'):
            os.remove('database.pkl')

    @staticmethod
    def mock_action(time, response, params, endpoint, result=np.nan):
        request_log = pd.DataFrame([{'time': time, 'params': params, 'endpoint': endpoint,
                                     'status': response.status_code, 'result': result}])
        pd.concat([pd.DataFrame(), request_log])

    def test_get_all_client_ids(self, mocked_responses):
        mocked_responses.get(URL_API_TEST + "client_ids",
                             json={"ids": [386902, 215797]},
                             status=200,
                             content_type="application/json")

        from dashboard import functions
        resp = functions.get_all_client_ids()
        assert resp == [386902, 215797]

    def test_get_numeric_features(self, mocked_responses):
        expected_content = ['DAYS_BIRTH', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY']
        mocked_responses.get(URL_API_TEST + 'numeric_features_list',
                             json={'numeric_features': expected_content},
                             status=200,
                             content_type='application/json')

        from dashboard import functions
        resp = functions.get_numeric_features()
        assert resp == expected_content

    def test_get_all_client_info_should_return_empty_dataframe(self, mocked_responses):
        mocked_responses.get(URL_API_TEST + "download_database",
                             json={"message": 'file not found'},
                             status=200,
                             content_type='application/json')

        from dashboard import functions
        resp = functions.get_all_client_info()
        assert isinstance(resp, pd.DataFrame)
        assert resp.empty

    def test_get_all_client_info_should_return_dataframe_not_empty(self, mocked_responses):
        mocked_responses.get(URL_API_TEST + "download_database",
                             body=open("api/data/X_sample.pkl", mode='rb').read(),
                             status=200,
                             content_type='application/pickle')

        from dashboard import functions
        resp = functions.get_all_client_info()
        assert isinstance(resp, pd.DataFrame)
        assert not resp.empty

    def test_get_default_threshold(self, mocked_responses):
        mocked_responses.get(URL_API_TEST + "threshold",
                             json={'threshold': 0.53},
                             status=200,
                             content_type='application/json')

        from dashboard import functions
        resp = functions.get_default_threshold()
        assert resp == 0.53

    def test_get_nearest_neighbors_ids(self, mocked_responses):
        expected_content = [215797, 386902, 309336, 233948, 176395, 265098]
        mocked_responses.get(URL_API_TEST + 'nearest_neighbors_ids',
                             json={"nearest_neighbors_ids": expected_content},
                             status=200,
                             content_type='application/json')

        from dashboard import functions
        resp = functions.get_nearest_neighbors_ids(215797, 5)
        assert resp == expected_content

    def test_get_client_default_proba(self, mocked_responses):
        mocked_responses.get(URL_API_TEST + "predict_default_one",
                             json={'prediction': "Crédit accordé", 'proba_default': 0.19456},
                             status=200,
                             content_type='application/json')

        from dashboard import functions
        resp = functions.get_client_default_proba_one(215797)
        assert resp['prediction'] == "Crédit accordé"
        assert resp['proba_default'] == 0.19456

    def test_get_all_clients_default_proba(self, mocked_responses):
        expected_content = [0.1945, 0.8976, 0.5643, 0.3452, 0.1237, 0.7843]
        mocked_responses.get(URL_API_TEST + 'predict_default_many',
                             json={'proba_default': expected_content},
                             status=200,
                             content_type='application/json')

        from dashboard import functions
        resp = functions.get_clients_default_proba_many([215797, 386902, 309336, 233948, 176395, 265098])
        assert resp == expected_content

    def test_get_client_info(self, mocked_responses):
        expected_content = {'NAME_CONTRACT_TYPE': "Cash loans",
                            'CODE_GENDER': 'F',
                            'CNT_CHILDREN': 0,
                            'AMT_INCOME_TOTAL': 112500,
                            'AMT_ANNUITY': 45369}
        mocked_responses.get(URL_API_TEST + "client_info",
                             json=[expected_content],
                             status=200,
                             content_type='application/json')

        from dashboard import functions
        resp = functions.get_client_info(215797)
        assert resp == expected_content

    def test_check_client_in_database(self, mocked_responses):
        mocked_responses.get(URL_API_TEST + "in_database",
                             json={"check": True},
                             status=200,
                             content_type='application/json')

        from dashboard import functions
        resp = functions.check_client_in_database(215797)
        assert resp

    # def test_build_waterfall_plot(self, mocked_responses):
        # content_json = {"shap_values": "[[-0.081963, 0.0, 0.0232220, 0.0164173, 0.0, 0.0]]",
                        # "expected_values": 0.005995,
                        # "features": ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALITY",
                                     # "NAME_CONTRACT_TYPE_Cash loans", "NAME_CONTRACT_TYPE_Revolving loans",
                                     # "NAME_TYPE_SUITE_Children"]
                        # }
        # mocked_responses.get(URL_API_TEST + "shap_values_default",
                             # json=content_json,
                             # status=200,
                             # content_type='application/json')

        # from dashboard import functions
        # resp = functions.build_waterfall_plot(215797)

    @pytest.mark.parametrize("amount", [2.6, 2_500.6, 2_500_500.6])
    def test_format_amount(self, amount):
        from dashboard import functions
        resp = functions.format_amount(amount)
        assert isinstance(resp, str)

        if amount == 2_500_500.6:
            assert resp == "2.5M"
        elif amount == 2_500.6:
            assert resp == "2.5k"
        elif amount == 2.6:
            assert resp == "3"
