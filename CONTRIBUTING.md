# Contributing to hypertiling

Thank you for considering a contribution!

## 1. Branching Strategy

We use a **feature-branch workflow**. The `main` branch is reserved for stable, tagged releases.

* **Target Branch:** All Pull Requests should target the **`dev`** branch.
* **Workflow:** 1. `git checkout dev`
    2. `git pull origin dev`
    3. `git checkout -b feature/your-feature-name`

## 2. Environment Setup

We use **Hatch** to manage our scientific Python stack and development tools.

1.  **Install Hatch:** `pip install hatch`
2.  **Initialize:** Run `hatch shell` in the project root.
    * *Note: Entering the shell automatically installs the **git pre-commit hooks**. These will silently strip Jupyter outputs and format your code whenever you commit.*



## 3. Testing

Before submitting a Pull Request, you **must** verify that your changes do not break existing tiling logic or the example gallery.

* **Unit Tests:** Run the core test suite:
    ```bash
    hatch run test:run
    ```
* **Notebook Verification:** Ensure all example notebooks still execute without errors:
    ```bash
    hatch run test:nb
    ```
* **Coverage:** To check if your new code is properly covered by tests:
    ```bash
    hatch run test:cov
    ```

## 6. Documentation

We use `nbsphinx` to render our examples into the official documentation. To ensure your docstrings and notebooks render correctly:
```bash
hatch run doc:build
```
Open `doc/_build/html/index.html` in your browser to preview the results.

---

### Why we require these steps:
* **Reproducibility:** Scientific code must produce consistent results across different environments. Hatch ensures we all use the same versions of `numpy`, `scipy`, and `numba`.
* **Repository Health:** Stripping notebook outputs prevents the `.git` folder from bloating with binary image data, keeping the repo fast for everyone.