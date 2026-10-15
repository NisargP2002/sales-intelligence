import importlib


def test_database_password_defaults_to_empty(monkeypatch):
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    import config

    reloaded = importlib.reload(config)
    assert reloaded.DB_PASSWORD == ""
    assert "sales_user:" in reloaded.DB_URL
