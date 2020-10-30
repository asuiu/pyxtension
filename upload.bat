del ./dist/*.whl
echo "Use --py2 option to build&upload py2 version"
python setup.py bdist_wheel %*
twine upload dist/*.whl -u asuiu --verbose