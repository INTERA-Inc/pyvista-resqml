import glob
import os
import shutil
import tarfile

from invoke import task

import meshio_resqml


@task
def build(c):
    shutil.rmtree("dist", ignore_errors=True)
    c.run("python -m build --sdist --wheel .")


@task
def html(c):
    c.run("sphinx-build -b html doc/source doc/build")


@task
def tag(c):
    c.run(f"git tag v{meshio_resqml.__version__}")
    c.run("git push --tags")


@task
def clean(c, bytecode=False):
    patterns = [
        ".pytest_cache",
        "build",
        "dist",
        "meshio_resqml.egg-info",
        "doc/build",
        "doc/source/examples",
    ]

    if bytecode:
        patterns += glob.glob("**/*.pyc", recursive=True)
        patterns += glob.glob("**/__pycache__", recursive=True)

    for pattern in patterns:
        if os.path.isfile(pattern):
            os.remove(pattern)
        else:
            shutil.rmtree(pattern, ignore_errors=True)


@task
def black(c):
    c.run("black -t py38 meshio_resqml")
    c.run("black -t py38 tests")


@task
def docstring(c):
    c.run("docformatter -r -i --blank --wrap-summaries 88 --wrap-descriptions 88 --pre-summary-newline meshio_resqml")


@task
def isort(c):
    c.run("isort meshio_resqml")
    c.run("isort tests")


@task
def format(c):
    c.run("invoke isort black docstring")


@task
def tar(c):
    patterns = [
        "__pycache__",
    ]

    def filter(filename):
        for pattern in patterns:
            if filename.name.endswith(pattern):
                return None
        
        return filename

    with tarfile.open("meshio_resqml.tar.gz", "w:gz") as tf:
        tf.add("meshio_resqml", arcname="meshio_resqml/meshio_resqml", filter=filter)
        tf.add("pyproject.toml", arcname="meshio_resqml/pyproject.toml")
        tf.add("setup.cfg", arcname="meshio_resqml/setup.cfg")
