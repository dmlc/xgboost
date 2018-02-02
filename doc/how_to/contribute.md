Contribute to XGBoost
=====================
XGBoost has been developed and used by a group of active community members.
Everyone is more than welcome to contribute. It is a way to make the project better and more accessible to more users.

- Please add your name to [CONTRIBUTORS.md](../../CONTRIBUTORS.md) after your patch has been merged.
- Please also update [NEWS.md](../../NEWS.md) to add note on your changes to the API or added a new document.

Guidelines
----------
* [Submit Pull Request](#submit-pull-request)
* [Git Workflow Howtos](#git-workflow-howtos)
  - [How to resolve conflict with master](#how-to-resolve-conflict-with-master)
  - [How to combine multiple commits into one](#how-to-combine-multiple-commits-into-one)
  - [What is the consequence of force push](#what-is-the-consequence-of-force-push)
* [Document](#document)
* [Testcases](#testcases)
* [Examples](#examples)
* [Core Library](#core-library)
* [Python Package](#python-package)
* [R Package](#r-package)

Submit Pull Request
-------------------
* Before submit, please rebase your code on the most recent version of master, you can do it by
```bash
git remote add upstream https://github.com/dmlc/xgboost
git fetch upstream
git rebase upstream/master
```
* If you have multiple small commits,
  it might be good to merge them together(use git rebase then squash) into more meaningful groups.
* Send the pull request!
  - Fix the problems reported by automatic checks
  - If you are contributing a new module, consider add a testcase in [tests](../tests)

Git Workflow Howtos
-------------------
### How to resolve conflict with master
- First rebase to most recent master
```bash
# The first two steps can be skipped after you do it once.
git remote add upstream https://github.com/dmlc/xgboost
git fetch upstream
git rebase upstream/master
```
- The git may show some conflicts it cannot merge, say ```conflicted.py```.
  - Manually modify the file to resolve the conflict.
  - After you resolved the conflict, mark it as resolved by
```bash
git add conflicted.py
```
- Then you can continue rebase by
```bash
git rebase --continue
```
- Finally push to your fork, you may need to force push here.
```bash
git push --force
```

### How to combine multiple commits into one
Sometimes we want to combine multiple commits, especially when later commits are only fixes to previous ones,
to create a PR with set of meaningful commits. You can do it by following steps.
- Before doing so, configure the default editor of git if you haven't done so before.
```bash
git config core.editor the-editor-you-like
```
- Assume we want to merge last 3 commits, type the following commands
```bash
git rebase -i HEAD~3
```
- It will pop up an text editor. Set the first commit as ```pick```, and change later ones to ```squash```.
- After you saved the file, it will pop up another text editor to ask you modify the combined commit message.
- Push the changes to your fork, you need to force push.
```bash
git push --force
```

### What is the consequence of force push
The previous two tips requires force push, this is because we altered the path of the commits.
It is fine to force push to your own fork, as long as the commits changed are only yours.

Documents
---------
* The document is created using sphinx and [recommonmark](http://recommonmark.readthedocs.org/en/latest/)
* You can build document locally to see the effect.

Testcases
---------
* All the testcases are in [tests](../tests)
* We use python nose for python test cases.

Examples
--------
* Usecases and examples will be in [demo](../demo)
* We are super excited to hear about your story, if you have blogposts,
  tutorials code solutions using xgboost, please tell us and we will add
  a link in the example pages.

Core Library
------------
- Follow Google C style for C++.
- We use doxygen to document all the interface code.
- You can reproduce the linter checks by typing ```make lint```

Python Package
--------------
- Always add docstring to the new functions in numpydoc format.
- You can reproduce the linter checks by typing ```make lint```

R Package
---------
### Code Style
- We follow Google's C++ Style guide on C++ code.
  - This is mainly to be consistent with the rest of the project.
  - Another reason is we will be able to check style automatically with a linter.
- You can check the style of the code by typing the following command at root folder.
```bash
make rcpplint
```
- When needed, you can disable the linter warning of certain line with ```// NOLINT(*)``` comments.
- We use [roxygen](https://cran.r-project.org/web/packages/roxygen2/vignettes/roxygen2.html) for documenting the R package.

### Rmarkdown Vignettes
Rmarkdown vignettes are placed in [R-package/vignettes](../R-package/vignettes)
These Rmarkdown files are not compiled. We host the compiled version on [doc/R-package](R-package)

The following steps are followed to add a new Rmarkdown vignettes:
- Add the original rmarkdown to ```R-package/vignettes```
- Modify ```doc/R-package/Makefile``` to add the markdown files to be build
- Clone the [dmlc/web-data](https://github.com/dmlc/web-data) repo to folder ```doc```
- Now type the following command on ```doc/R-package```
```bash
make the-markdown-to-make.md
```
- This will generate the markdown, as well as the figures into ```doc/web-data/xgboost/knitr```
- Modify the ```doc/R-package/index.md``` to point to the generated markdown.
- Add the generated figure to the ```dmlc/web-data``` repo.
  - If you already cloned the repo to doc, this means a ```git add```
- Create PR for both the markdown  and ```dmlc/web-data```
- You can also build the document locally by typing the following command at ```doc```
```bash
make html
```
The reason we do this is to avoid exploded repo size due to generated images sizes.

### R package versioning
Since version 0.6.4.3, we have adopted a versioning system that uses an ```x.y.z``` (or ```core_major.core_minor.cran_release```)
format for CRAN releases and an ```x.y.z.p``` (or ```core_major.core_minor.cran_release.patch```) format for development patch versions.
This approach is similar to the one described in Yihui Xie's
[blog post on R Package Versioning](https://yihui.name/en/2013/06/r-package-versioning/),
except we need an additional field to accomodate the ```x.y``` core library version.

Each new CRAN release bumps up the 3rd field, while developments in-between CRAN releases
would be marked by an additional 4th field on the top of an existing CRAN release version.
Some additional consideration is needed when the core library version changes.
E.g., after the core changes from 0.6 to 0.7, the R package development version would become 0.7.0.1, working towards
a 0.7.1 CRAN release. The 0.7.0 would not be released to CRAN, unless it would require almost no additional development.

### Registering native routines in R
According to [R extension manual](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Registering-native-routines),
it is good practice to register native routines and to disable symbol search. When any changes or additions are made to the
C++ interface of the R package, please make corresponding changes in ```src/init.c``` as well.
