<p align="left">
  <img src="docs/_static/logo/daspal_logo.png" alt="Daspal logo" width="60%">
</p>

**DAS Processing & Analysis Library**

[![PyPI version](https://img.shields.io/pypi/v/daspal.svg)](https://pypi.org/project/daspal/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License](https://img.shields.io/pypi/l/daspal.svg)](LICENSE.txt)
<!-- [![Documentation Status](https://readthedocs.org/projects/daspal/badge/?version=latest)](https://daspal.readthedocs.io/en/latest/) -->

---

## Description

This python project presents a set of tools for **Distibuted Acoustic Sensing (DAS) data processing and analysis**. It aims to leverage on existing widely used python libraries within the geoscience community, mainly [Xarray](https://docs.xarray.dev/en/stable/index.html) and [Dask](https://docs.dask.org/en/stable/), for efficient reading, processing and visualization of DAS data. `daspal` offers easily scalable DAS data analysis tools that be ran on simple laptop and HPC clusters, even cloud computing environments with Xarray and Dask integrated compatibility with could-native data formats such as [Zarr](https://zarr.readthedocs.io/en/stable/).

The package's documentation can be found [here](https://daspal.readthedocs.io/en/stable/).

---

## Installation

Install `daspal` using **pip**:

```bash
pip install daspal
```

---

## Examples

In order to get familiar with some of the modules to read, process and visualize DAS data, example notebooks (based on OptoDAS data) are available in the [`docs/examples`](https://github.com/daspal-project/daspal/tree/master/docs/examples) folder:

* [`Reading_and_processing_data.ipynb`](https://github.com/daspal-project/daspal/tree/master/docs/examples/Reading_and_processing_data.ipynb)
* [`Lazy_data_pipeline.ipynb`](https://github.com/daspal-project/daspal/tree/master/docs/examples/Lazy_data_pipeline.ipynb)
* [`Channel_spectrograms.ipynb`](https://github.com/daspal-project/daspal/tree/master/docs/examples/Channel_spectrograms.ipynb)
* [`Working_with_Zarr_data_format.ipynb`](https://github.com/daspal-project/daspal/tree/master/docs/examples/Working_with_Zarr_data_format.ipynb)

The notebooks are also rendered in the online documentation [here](https://daspal.readthedocs.io/en/stable/examples/).

The access to the data used by the notebooks is described and accessible [here](https://daspal.readthedocs.io/en/stable/getting_started.html#getting-the-sample-data).


---

## License

`daspal` is distributed under the terms of the [GPL-3.0-or-late](https://spdx.org/licenses/GPL-3.0-or-later.html) license.

---

## Contributing

We welcome contributions of all kinds: bug fixes, new features, or improvements to documentation.

For more information, see our [development page](https://daspal.readthedocs.io/en/stable/development.html).

---

## AI-assisted development

This project was designed and implemented by the authors. However, LLMs were used along the way as an interactive assistant for code suggestions, boilerplate generation, and prototyping. The authors take full responsibility for the content of the codebase.

---

## Developers

This package is developed and maintained by:

* **Florian Le Pape** – [florian.le.pape@ifremer.fr](mailto:florian.le.pape@ifremer.fr)
* **Gwenaël Caër** – [gwenael.caer@data-terra.org](mailto:gwenael.caer@data-terra.org)
