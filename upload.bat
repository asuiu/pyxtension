python setup.py sdist bdist_wheel
echo "ASU: just copy ./py2 and ./py3 directories into generated dist/*.tar.gz, and don't forget to bump the version, exclude .idea & tests"
echo "ASU: Wheel for Py3 works"
pause "Press enter to continue"
twine upload dist/* --verbose