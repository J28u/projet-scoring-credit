import pytest
import responses
from fastapi.testclient import TestClient


@pytest.fixture(scope='session')
def monkey_session():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope='session')
def client(monkey_session):
    monkey_session.setenv('DATA_PATH', 'api/data')
    monkey_session.setenv('MODEL_PATH', 'api/models')

    from api.app.main import App
    app_test = App('AppTest', 'App to test functions', '0.1').create_app()

    with TestClient(app_test) as client:
        yield client


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps:
        yield rsps
