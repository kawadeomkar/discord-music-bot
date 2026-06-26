import os
from enum import Enum


class Environment(str, Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"


def _parse() -> Environment:
    raw = os.getenv("ENVIRONMENT", "development").lower()
    match raw:
        case "production":
            return Environment.PRODUCTION
        case "staging":
            return Environment.STAGING
        case _:
            return Environment.DEVELOPMENT


ENVIRONMENT = _parse()
