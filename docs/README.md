# Compiling protex's Documentation

## General
The documentation is currently (17.08.2026) built via GitHub Pages from the `/root` folder of the `docs` branch and needs to be added manually. Make sure to pull/merge regularly to include new functions etc. in the autosummaries. Also, the autosummary only works if we keep the code up to date with nice docstrings.

Flo had a bot to update the autosummaries etc. automatically, but it fails now because GitHub deprecated some stuff. If anyone gets it running again, so much the better, but I gave up.

## Compiling
The docs are built with [Sphinx](http://www.sphinx-doc.org/en/master/).
To compile the docs, first ensure that Sphinx and the ReadTheDocs theme are installed.


```bash
conda install sphinx sphinx_rtd_theme 
```


Once installed, you can use the `Makefile` in this directory to compile static HTML pages by
```bash
cd docs
make html
```

The compiled docs will be in the `_build` directory and can be viewed by opening `index.html` (which may itself 
be inside a directory called `html/` depending on what version of Sphinx is installed).

The way I can make it work now is to copy the contents of `_build/htmml` to `/root`, commit, and push, since I can't choose `docs/_build/html` on GitHub. There is possibly a way to change the directories in the index so that it finds the rest of the files. There has to be a `.nojekyll` file as well; otherwise GitHub gets confused with the styles. 

```bash
cd ..
touch .nojekyll
cp -r /docs/_build/html* .
git add .
git commit -m "update documentation"
git push origin docs
```

## Hosting

### GitHub Pages
This is how we do it currently.

Set the branch and folder on GitHub: Settings -> Pages. `/root` or `/root/docs` can be used. There needs to be a README.md, index.html, or similar in that folder. 

### ReadTheDocs
A configuration file for [Read The Docs](https://readthedocs.org/) (readthedocs.yaml) is included in the top level of the repository. To use Read the Docs to host your documentation, go to https://readthedocs.org/ and connect this repository. You may need to change your default branch to `main` under Advanced Settings for the project.

If you would like to use Read The Docs with `autodoc` (included automatically) and your package has dependencies, you will need to include those dependencies in your documentation yaml file (`docs/requirements.yaml`).

