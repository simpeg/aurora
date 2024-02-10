---
title: 'Aurora: An open-source python implementation of the EMTF package for magnetotelluric data processing using MTH5 and mt_metadata'
tags:
  - Python
  - Geophysics
  - Magnetotellurics
  - Time series
authors:
  - name: Karl N. Kappler
    orcid: 0000-0002-1877-1255
    affiliation: 5
  - name: Jared R. Peacock
    orcid: 0000-0002-0439-0224F
    affiliation: 1
  - name: Gary D. Egbert
    orcid: 0000-0003-1276-8538
    affiliation: 2
  - name: Andrew Frassetto
    orcid: 0000-0002-8818-3731
    affiliation: 3
  - name: Lindsey Heagy
    orcid: 0000-0002-1551-5926
    affiliation: 4
  - name: Anna Kelbert
    orcid: 0000-0003-4395-398X
    affiliation: 1
  - name: Laura Keyson
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Douglas Oldenburg
    orcid: 0000-0002-4327-2124
    affiliation: 4
  - name: Timothy Ronan
    orcid: 0000-0001-8450-9573
    affiliation: 3
  - name: Justin Sweet
    orcid: 0000-0001-7323-9758
    affiliation: 3
affiliations:
 - name: United States Geological Survey, USA
   index: 1
 - name: Oregon State University, USA
   index: 2
 - name: Earthscope, USA
   index: 3
 - name: University of British Columbia, USA
   index: 4
 - name: Independent Researcher, USA
   index: 5
date: 12 January 2023
bibliography: paper.bib
---
# Summary

The Aurora software package robustly estimates single station and remote reference electromagnetic transfer functions (TFs) from magnetotelluric (MT) time series.  Aurora is part of an open-source processsing workflow that leverages the self-describing data container MTH5, which in turn leverages the general mt\_metadata framework to manage metadata.  These pre-existing tools greatly simplify the processing interface, reducing requirements for specialized domain knowledge in time series analysis, or data structures manangement and generating transfer functions with a few lines of code.  The processing depends on two inputs -- a table specifying the data to use for TF estimation, and a JSON file specfiying the processing parameters, both of which are generated automatically, and can be modified if desired.  Output TFs are returned as mt\_metadata objects, and can be exported to a variety of common formats for plotting, modelling and inversion.  

# Introduction

Magnetotellurics (MT) is a geophysical technique for probing the electrical conductivity structure of the subsurface using co-located electric and magnetic field measurements.  After data collection, standard practice is to estimate the time invariant  (frequency domain) transfer function (TF) between electric and magnetic channels before proceeding to interpretation and modelling. If measurements are orthogonal the TF is equivlent to the electrical impedance tensor (Z) [@Vozoff:1991].  


$\begin{bmatrix} E_x \\ E_y \end{bmatrix}
=
\begin{bmatrix} 
Z_{xx} & Z_{xy} \\ 
Z_{yx} & Z_{yy} 
\end{bmatrix}
\begin{bmatrix} H_x \\ H_y \end{bmatrix}$
 
where ($E_x$, $E_y$), ($H_x$, $H_y$) denote orthogonal electric and magnetic fields respectively.  TF estimation involves management of metadata (locations, orientations, timestamps,) versatile data containers (for linear algebra, slicing, plotting, etc.) and uses a broad collection of signal processing and statistical techniques (@egbert1997robust and references therein).  MTH5 supplies time series as xarray objects for efficient, lazy access to data and easy application of linear algebra and statistics libraries available in the python.

# Statement of Need

Uncompiled FORTRAN processing codes have been available for years (e.g. EMTF @egbert2017mod3dmt, or BIRRP @chave1989birrp) but do not offer the readability of a high-level languge and modifications are seldom attempted [@egbert2017mod3dmt]. Recently several python versions of MT processing codes have been released by the open source community, including @shah2019resistics, @smai2020razorback, @ajithabh2023sigmt, and @mthotel.  Aurora adds to this canon of options but differs by leveraging the MTH5 and mt\_metadata packages elimiating a need for internal development of time series or metadata containers.  By providing an example workflow employing mt\_metadata and MTH5 we hope other developers may benefit from following this model, allowing researchers interested in signal-and-noise separation in MT to spend more time exploring and testing algorithms to improve TF estimates, and less time (re)-developing formats and management tools for data and metadata. As a python representation of Egbert's EMTF Remote Reference processing software, Aurora also provides a sort of continuity in the code space as the languages evolve.  

This manuscript describes the high-level concepts of the software – for information about MT data processing @ajithabh2023sigmt provides a concise summary, and more in-depth details can be found in @Vozoff:1991, @egbert2002processing and references therein.  


# Key Features

- Tabular Data indexing and management (pandas data frames), 
- Dicitionary-like Processing parameters configuration
- Both allow for programatic or manual editting.


A TF instance depends on two key prior decisions: a) The data input to the TF computation algorithm, b) The algorithm itself including the specific values of the processing parameters.  Aurora formalizes these concepts as classes (KernelDataset and Processing, respectively), and a third class TransferFunctionKernel (TFK Figure \ref{TFK}), a compositon of the Processing, and KernelDataset provides a place for logic validating consistency between selected data and processing parameters. TFK specifies all the information needed to make the calculation of a TF reproducible (supporting the R in FAIRly archived TFs).

Generation of robust TFs can be done in only a few lines starting from an MTH5 archive (Figure \ref{minimal_example}).  Simplicity of workflow is due to the MTH5 container already storing comprehensive metadata, including a channel summary table describing all time series stored in the archive including start/end times and sample rates.  Users can easily view a tabular summary of available data and select station pairs to process.  Once a station -- and optionally a remote reference station -- are defined, the simultaneous time intervals of data coverage at both stations are idenitified automatically, providing the Kernel Dataset.  Reasonable starting processing parameters can be automatically generated for a given Kernel Dataset, and editted programatically or via a JSON file. Once the TFK is defined, the processing automatically follows the flow of Figure \ref{FLOW}.

![TF Kernel concept diagram: A box representing the TF Kernel with two inlay boxes representing the processing config (JSON) and dataset (pandas DataFrame).  \label{TFK}](TFKernelApproach.png)

![The main interfaces of Aurora TF processing. \label{FLOW}  Example time series from mth5 archive in linked notebook (using MTH5 built-in time series plotting), spectrogram from FC data structure, cartoon from @hand2018statistical and TF from [SPUD](https://ds.iris.edu/spud/emtf/18633652).](aurora_workflow.png)

# Examples

Here an example of the aurora data processing flow is given, using data from Earthscope.  This section refers to Jupyter notebooks intended as companions to this paper. A relatively general notebook about accessing Earthscope data with MTH5 can be found in the link from row 1 of Table \ref{jupyter}.

Table: \label{jupyter} Referenced jupyter notebooks with links.

| ID | Link |
|--|:---|
| 1 | [earthscope_magnetic_data_tutorial](https://github.com/simpeg/aurora/blob/patches/docs/examples/earthscope_magnetic_data_tutorial.ipynb) |
| 2 | [make_mth5_driver_v0.2.0](https://github.com/kujaku11/mth5/blob/master/docs/examples/notebooks/make_mth5_driver_v0.2.0.ipynb) |
| 3 | [process_cas04_mulitple_station](https://github.com/simpeg/aurora/blob/joss/docs/examples/process_cas04_mulitple_station.ipynb) |


The MTH5 dataset can be built by executing the example notebook in row 2 of Table \ref{jupyter}.The data processing can be executed by following the tutorial in row 3 of Table \ref{jupyter} -- a condensed version of which is shown in Figure \ref{minimal_example}.  Resultant apparent resistivities are plotted in Figure \ref{compareTFs} along with the results hosted at Earthscope from EMTF.  

![Code snippet with steps to generate a TF from an MTH5 (generated by row 1 of Table \ref{jupyter}).  With MTH5 in present working directory, a table of available contiguous blocks of multichannel time series is generated ("RunSummary"), then station(s) to process are selected (by inspection of the RunSummary dataframe) to generate a KernelDataset.  The KernelDataset identifies simultaneous data at the local and reference site, and generates processing parameters, which can be editted before passing them to process_mth5, and finally exporting TF to a standard output format, in this case `edi`. \label{minimal_example}](processing_code_example.png)

![Comparison of apparent resistivities from Aurora and EMTF for station CAS04.  Both curves exhibit scatter in the low SNR MT "dead band" beween 1-10s, but most of estimates are very similar. \label{compareTFs}](tf_comparison.png)


# Testing
The Aurora package uses continuous integration [@duvall2007continuous] and implements both unit tests as well as integrated tests with 77% code coverage as measured by CodeCov (core dependencies mt_metadata and MTH5 at 84% and 60% respectively).  Improvement of test coverage is ongoing.  For integreated tests Aurora uses a small synthetic MT dataset originally from EMTF.  A few processing configurations with manually validated results are stored in the repository. Deviation from these results causes tests to fail, alerting developers if a code change resulted in an unexpected baseline processing result.  In the summer of 2023, widescale testing on Earthscope data archives was performed and showed that the TF results of auora are similar to those form the EMTF fortran codes, in this case for hundreds of real stations rather than a few synthetic ones.  Before PyPI, and conda forge releases, example Jupyter notebooks are also run via github actions.


# Future Work
Aurora uses github issues to track tasks and planned improvments.  In the near future we want to add noise suppression techniques, for example coherence and polarization sorting and Mahalanobis distance (e.g. @ajithabh2023sigmt, @platz2019automated).  We would also like to develop, or plug into a graphical data selection/rejection interface with time series plotting. Besides these improvments to TF quality, we also would like to embed the TFKernel information into both the MTH5 and the output EMTF\_XML (@kelbert2020emtf). Unit and integrated tests should be expanded, including a test dataset from audio MT band (most test data is sampled at 1Hz). Aurora will continue to codevelop with mt\_metadata, MTH5 and MTPy to maintain the ability to provide outputs for inversion and modelling. Ideally the community can participate in a comparative analysis of the opensource codes available to build a recipe book for handling noise from various open-archived datasets.


# Conclusion
Aurora provides an open-source Python implementation of the EMTF package for magnetotelluric data processing.  Aurora is a prototype worked example of how to plug processing into an existing opensource data and metadata ecosystem (MTH5, mt_metadata, & MTpy).  We hope Aurora can be used as an example interface to these packages for the open source MT community, and that these tools will contribute to workflows which can focus more on geoscience analysis, and less on the nuances of data management. 


# Appendix

## Installation Instructions
The package is installable via the Python Package Index (PyPI) as well as via conda-forge.

The installation in pip:
	> pip install aurora

And via conda forge:
	> conda install aurora


## Documentation
Documentation is hosted by SimPEG @cockett2015simpeg and can be found at this [link](http://simpeg.xyz/aurora/)


## Licence
Aurora is distributed under the [MIT](https://opensource.org/license/mit/) open-source licence.


# Acknowledgments 
The authors would like to thank IRIS (now Earthscope) for supporting the development of Aurora.  Joe Capriotti at SimPEG helped with online documentation and the initial release.


TODO:

- [ ] **Update links to ipynb to release branches after mth5/aurora releases**.

- [ ] remove draft watermark

- [ ] Link these issues to discussion in future work? https://github.com/kujaku11/mth5/issues/179, https://github.com/kujaku11/mt_metadata/issues/195

```python

```