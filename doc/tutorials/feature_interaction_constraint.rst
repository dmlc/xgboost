###############################
Feature Interaction Constraints
###############################

The decision tree is a powerful tool to discover interaction among independent
variables (features). Variables that appear together in a traversal path
are interacting with one another, since the condition of a child node is
predicated on the condition of the parent node. For example, the highlighted
red path in the diagram below contains three variables: :math:`x_1`, :math:`x_7`,
and :math:`x_{10}`, so the highlighted prediction (at the highlighted leaf node)
is the product of interaction between :math:`x_1`, :math:`x_7`, and
:math:`x_{10}`.

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph feature_interaction_illustration1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label=<x<SUB><FONT POINT-SIZE="11">10</FONT></SUB> &lt; -1.5 ?>, shape=box, color=red, fontcolor=red];
      1 [label=<x<SUB><FONT POINT-SIZE="11">2</FONT></SUB> &lt; 2 ?>, shape=box];
      2 [label=<x<SUB><FONT POINT-SIZE="11">7</FONT></SUB> &lt; 0.3 ?>, shape=box, color=red, fontcolor=red];
      3 [label="...", shape=none];
      4 [label="...", shape=none];
      5 [label=<x<SUB><FONT POINT-SIZE="11">1</FONT></SUB> &lt; 0.5 ?>, shape=box, color=red, fontcolor=red];
      6 [label="...", shape=none];
      7 [label="...", shape=none];
      8 [label="Predict +1.3", color=red, fontcolor=red];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      0 -> 2 [labeldistance=2.0, labelangle=-45,
              headlabel="No", color=red, fontcolor=red];
      1 -> 3 [labeldistance=2.0, labelangle=45, headlabel="Yes"];
      1 -> 4 [labeldistance=2.0, labelangle=-45, headlabel="             No/Missing"];
      2 -> 5 [labeldistance=2.0, labelangle=-45, headlabel="Yes",
              color=red, fontcolor=red];
      2 -> 6 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
      5 -> 7;
      5 -> 8 [color=red];
    }
  """
  Source(source, format='png').render('../_static/feature_interaction_illustration1', view=False)
  Source(source, format='svg').render('../_static/feature_interaction_illustration1', view=False)

.. raw:: html

  <p>
  <img src="../_static/feature_interaction_illustration1.svg"
    onerror="this.src='../_static/feature_interaction_illustration1.png'; this.onerror=null;">
  </p>

When the tree depth is larger than one, many variables interact on
the sole basis of minimizing training loss, and the resulting decision tree may
capture a spurious relationship (noise) rather than a legitimate relationship
that generalizes across different datasets. **Feature interaction constraints**
allow users to decide which variables are allowed to interact and which are not.

Potential benefits include:

* Better predictive performance from focusing on interactions that work --
  whether through domain specific knowledge or algorithms that rank interactions
* Less noise in predictions; better generalization
* More control to the user on what the model can fit. For example, the user may
  want to exclude some interactions even if they perform well due to regulatory
  constraints

****************
A Simple Example
****************

Feature interaction constraints are expressed in terms of groups of variables
that are allowed to interact. For example, the constraint
``[0, 1]`` indicates that variables :math:`x_0` and :math:`x_1` are allowed to
interact with each other but with no other variable. Similarly, ``[2, 3, 4]``
indicates that :math:`x_2`, :math:`x_3`, and :math:`x_4` are allowed to
interact with one another but with no other variable. A set of feature
interaction constraints is expressed as a nested list, e.g.
``[[0, 1], [2, 3, 4]]``, where each inner list is a group of indices of features
that are allowed to interact with each other.

In the following diagram, the left decision tree is in violation of the first
constraint (``[0, 1]``), whereas the right decision tree complies with both the
first and second constraints (``[0, 1]``, ``[2, 3, 4]``).

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph feature_interaction_illustration2 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label=<x<SUB><FONT POINT-SIZE="11">0</FONT></SUB> &lt; 5.0 ?>, shape=box];
      1 [label=<x<SUB><FONT POINT-SIZE="11">2</FONT></SUB> &lt; -3.0 ?>, shape=box];
      2 [label="+0.6"];
      3 [label="-0.4"];
      4 [label="+1.2"];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="No"];
      1 -> 3 [labeldistance=2.0, labelangle=45, headlabel="Yes"];
      1 -> 4 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
    }
  """
  Source(source, format='png').render('../_static/feature_interaction_illustration2', view=False)
  Source(source, format='svg').render('../_static/feature_interaction_illustration2', view=False)

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph feature_interaction_illustration3 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label=<x<SUB><FONT POINT-SIZE="11">3</FONT></SUB> &lt; 2.5 ?>, shape=box];
      1 [label="+1.6"];
      2 [label=<x<SUB><FONT POINT-SIZE="11">2</FONT></SUB> &lt; -1.2 ?>, shape=box];
      3 [label="+0.1"];
      4 [label="-0.3"];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes"];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
      2 -> 3 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      2 -> 4 [labeldistance=2.0, labelangle=-45, headlabel="No"];
    }
  """
  Source(source, format='png').render('../_static/feature_interaction_illustration3', view=False)
  Source(source, format='svg').render('../_static/feature_interaction_illustration3', view=False)

.. raw:: html

  <p>
  <img src="../_static/feature_interaction_illustration2.svg"
       onerror="this.src='../_static/feature_interaction_illustration2.png'; this.onerror=null;">
  <img src="../_static/feature_interaction_illustration3.svg"
       onerror="this.src='../_static/feature_interaction_illustration3.png'; this.onerror=null;">
  </p>

****************************************************
Enforcing Feature Interaction Constraints in XGBoost
****************************************************

It is very simple to enforce monotonicity constraints in XGBoost.  Here we will
give an example using Python, but the same general idea generalizes to other
platforms.

Suppose the following code fits your model without monotonicity constraints:

.. code-block:: python

  model_no_constraints = xgb.train(params, dtrain,
                                   num_boost_round = 1000, evals = evallist,
                                   early_stopping_rounds = 10)

Then fitting with monotonicity constraints only requires adding a single
parameter:

.. code-block:: python

  params_constrained = params.copy()
  # Use nested list to define feature interaction constraints
  params_constrained['interaction_constraints'] = '[[0, 2], [1, 3, 4], [5, 6]]'
  # Features 0 and 2 are allowed to interact with each other but with no other feature
  # Features 1, 3, 4 are allowed to interact with one another but with no other feature
  # Features 5 and 6 are allowed to interact with each other but with no other feature

  model_with_constraints = xgb.train(params_constrained, dtrain,
                                     num_boost_round = 1000, evals = evallist,
                                     early_stopping_rounds = 10)

**Choice of tree construction algorithm**. To use feature interaction
constraints, be sure to set the ``tree_method`` parameter to either ``exact``
or ``hist``. Currently, GPU algorithms (``gpu_hist``, ``gpu_exact``) do not
support feature interaction constraints.
