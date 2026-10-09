# Getting Started

`daspal` has been developped to create a set of modules for reading, processing and visualization of Distributed Acoustic Sensing (DAS) data. With Xarray and Dask at the core of the software, the user can get multiple out of the box exploitable features that offer seamless scalability on DAS data analysis.

## Installation

`daspal` is a pure Python package and can be installed using `pip`.

Install the latest version from PyPI:
```bash
pip install daspal
```

### Development versions

To install the latest development version from source:

```bash
python -m pip install git+https://github.com/daspal-project/daspal.git@devel
```

## DAS data sample access

DAS data to be used with the notebooks is described here : [OMAC 10 min DAS data sample](https://sextant.ifremer.fr/Donnees/Catalogue#/metadata/85a5cfc4-4bec-4f0a-97c4-d6fed88d9fb3), and can be directly accessed via the command below:

```bash
wget -r -np -nH "https://data-sextant.ifremer.fr/OMAC_10min_sample/"
```

Then modify in the notebooks the path to the data accordingly.
