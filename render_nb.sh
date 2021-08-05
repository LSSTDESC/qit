jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute nb/qit_binned.ipynb; \mv nb/qit_binned.html docs
jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute nb/qit_example.ipynb; \mv nb/qit_example.html docs
