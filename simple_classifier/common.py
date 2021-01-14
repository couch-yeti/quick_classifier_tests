"""Contains common helper functions."""
import newrelic.agent
import sys
import os


# globals
local = "win" in sys.platform


def newrelic_wrapper(func):
    is_local = local
    if is_local:
        return func
    newrelic.agent.initialize()
    return newrelic.agent.lambda_handler()(func)
