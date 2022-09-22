.. _donation_policy:

Donations
=========

.. raw:: html

  <a href="https://opencollective.com/xgboost">Donate to dmlc/xgboost</a>

Motivation
----------
DMLC/XGBoost has grown from a research project incubated in academia to one of the most widely used gradient boosting framework in production environment. On one side, with the growth of volume and variety of data in the production environment, users are putting accordingly growing expectation to XGBoost in terms of more functions, scalability and robustness. On the other side, as an open source project which develops in a fast pace, XGBoost has been receiving contributions from many individuals and organizations around the world. Given the high expectation from the users and the increasing channels of contribution to the project, delivering the high quality software presents a challenge to the project maintainers.

A robust and efficient **continuous integration (CI)** infrastructure is one of the most critical solutions to address the above challenge. A CI service will monitor an open-source repository and run a suite of integration tests for every incoming contribution. This way, the CI ensures that every proposed change in the codebase is compatible with existing functionalities. Furthermore, XGBoost can enable more thorough tests with a powerful CI infrastructure to cover cases which are closer to the production environment.

There are several CI services available free to open source projects, such as Travis CI and AppVeyor. The XGBoost project already utilizes GitHub Actions. However, the XGBoost project has needs that these free services do not adequately address. In particular, the limited usage quota of resources such as CPU and memory leaves XGBoost developers unable to bring "too-intensive" tests. In addition, they do not offer test machines with GPUs for testing XGBoost-GPU code base which has been attracting more and more interest across many organizations. Consequently, the XGBoost project uses a cloud-hosted test farm. We use `BuildKite <https://buildkite.com/xgboost>`_ to organize CI pipelines.

The cloud-hosted test farm has recurring operating expenses. It utilizes a leading cloud provider (AWS) to accommodate variable workload. BuildKite launches worker machines on AWS on demand, to run the test suite on incoming contributions. To save cost, the worker machines are terminated when they are no longer needed.

To help defray the hosting cost, the XGBoost project seeks donations from third parties.

Donations and Sponsorships
--------------------------
Donors may choose to make one-time donations or recurring donations on monthly or yearly basis. Donors who commit to the Sponsor tier will have their logo displayed on the front page of the XGBoost project.

Fiscal host: Open Source Collective 501(c)(6)
---------------------------------------------
The Project Management Committee (PMC) of the XGBoost project appointed `Open Source Collective <https://opencollective.com/opensource>`_ as their **fiscal host**. The platform is a 501(c)(6) registered entity and will manage the funds on the behalf of the PMC so that PMC members will not have to manage the funds directly. The platform currently hosts several well-known JavaScript frameworks such as Babel, Vue, and Webpack.

All expenses incurred for hosting CI will be submitted to the fiscal host with receipts. Only the expenses in the following categories will be approved for reimbursement:

* Cloud exprenses for the cloud test farm (https://buildkite.com/xgboost)
* Cost of domain https://xgboost-ci.net
* Monthly cost of using BuildKite
* Hosting cost of the User Forum (https://discuss.xgboost.ai)

Administration of cloud CI infrastructure
-----------------------------------------
The PMC shall appoint committer(s) to administer the cloud CI infrastructure on their behalf. The current administrators are as follows:

* Primary administrator: `Hyunsu Cho <https://github.com/hcho3>`_
* Secondary administrator: `Jiaming Yuan <https://github.com/trivialfis>`_

The administrators shall make good-faith effort to keep the CI expenses under control. The expenses shall not exceed the available funds. The administrators should post regular updates on CI expenses.
