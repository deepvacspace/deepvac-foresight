from flask import Flask


def create_app(config_class=None):
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY="deepvac-dashboard-dev",
        JSON_SORT_KEYS=False,
    )

    if config_class:
        app.config.from_object(config_class)

    from app.core.views import core

    app.register_blueprint(core)
    return app
