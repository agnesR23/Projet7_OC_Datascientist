#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:48:07 2023

@author: agnes
"""


        
        
        
        
import pytest
from API import app as flask_app
@pytest.fixture
def app():
    yield flask_app
@pytest.fixture
def client(app):
    return app.test_client()