#!/usr/bin/env bash
set -e

function has_command() {
  type $1 &> /dev/null
}

if has_command apt-get; then
  echo 'Performing Linux Specific Setup'
  sudo apt-get install git-lfs git-extras swig -y
fi

if has_command port; then
  echo 'Performing Mac Specific Setup'
  sudo brew install git-extras
fi

pip install -r requirements-dev.txt

if test -e .git/hooks/pre-commit; then
  echo 'pre-commit hook already exists, skipping'
else
  echo 'Installing git pre-commit hook'
  pre-commit install
  pre-commit autoupdate
fi

git config push.recurseSubmodules on-demand

# if on Windows, to make pre-commit hook work on PyCharm,
# see next instructions: https://stackoverflow.com/a/68293234/691343