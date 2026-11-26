# mytools

A small personal Python library that prints the practical code, simple pseudocode,
and procedure steps for Practicals 1–7 in the supplied *Foundations of Artificial
Intelligence and Machine Learning* lab manual.

## Folder structure

```text
mytools/
│
├── mytools/
│   ├── __init__.py
│   └── functions.py
│
└── README.md
```

## Install from GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bash
pip install git+https://github.com/YOUR_USERNAME/mytools.git
```

The repository must be public for this command to work without GitHub authentication.

## Use

```python
import mytools

mytools.getpract()

# Print practical code
mytools.linearRegression()
mytools.logisticRegression()
mytools.Clusters()
mytools.ANN()
mytools.Activation()
mytools.Vaccum()
mytools.Traffic()
mytools.Genetic()

# Print pseudocode
mytools.Clusters_pseudo()

# Print procedure steps
mytools.Genetic_steps()
```

You can also import individual functions:

```python
from mytools import linearRegression, linearRegression_pseudo
linearRegression()
linearRegression_pseudo()
```

## Notes

- The functions print code as text; they do not execute the practical automatically.
- Practical 1 is split into `linearRegression()` and `logisticRegression()`.
- The function spelling `Vaccum` is kept exactly as requested.
- Practical code follows the manual's datasets, parameters, and workflow. Indentation
  and obvious formatting issues in the printed code have been normalized so the
  snippets are valid Python.
- To run a printed practical, copy its displayed code into a notebook/script and
  install the dependencies it imports (for example, NumPy, Matplotlib, and
  scikit-learn).
