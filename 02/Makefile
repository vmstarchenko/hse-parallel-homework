src_dir=src
tests_dir=tests


clean:
	find . -name \*.pyc -delete
	find . -name __pycache__ -delete
	find . -name ".mypy_cache" -exec rm -rf {} \;
	find . -name "flycheck_*" -exec rm -rf {} \;


utest:
	python -m unittest ${tests_dir} --verbose


lint:
	( pylint ${src_dir} --rcfile=etc/pylint/.pylintrc --ignore-patterns=flycheck_* \
	|| true )

lint_tests:
	( pylint ${tests_dir} --rcfile=etc/pylint/.pylintrc --ignore-patterns=flycheck_* --disable=missing-docstring,invalid-name \
	|| true )


format:
	pyformat -i -r ${src_dir}


format_tests:
	pyformat -i -r ${tests_dir}


tree: clean
	tree -I "venv|tmp|*.egg-info" .


# venv
new_venv: remove_venv
	python3.6 -m venv venv --copies


remove_venv:
	rm -rf venv


install:
	pip install -r requirements.txt


freeze:
	pip freeze > requirements.txt



