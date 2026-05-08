"""
Structured logging configuration using structlog.

Outputs JSON in production and pretty-printed console logs in development.
Import and call setup_logging() once at app startup.
"""
import logging
import os

import structlog


def setup_logging() -> None:
    env = os.getenv("ENVIRONMENT", "development")

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if env == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Mirror to stdlib so uvicorn / third-party logs are captured
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if env != "development" else logging.DEBUG,
    )
