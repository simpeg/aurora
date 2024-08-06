---
title: 'Aurora: An open-source Python implementation of the EMTF package for magnetotelluric data processing using MTH5 and mt_metadata'
tags:
  - Python
  - Geophysics
  - Magnetotellurics
  - Time series
authors:
  - name: Karl N. Kappler
    orcid: 0000-0002-1877-1255
    affiliation: "5,6"
  - name: Jared R. Peacock
    orcid: 0000-0002-0439-0224
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
 - name: U.S. Geological Survey, USA
   index: 1
 - name: Oregon State University, USA
   index: 2
 - name: EarthScope, USA
   index: 3
 - name: University of British Columbia, USA
   index: 4
 - name: Space Science Institute, USA
   index: 5
 - name: DIAS Geophysical, Canada
   index: 6
date: 12 January 2023
bibliography: paper.bib
---

# Summary

The Aurora software package robustly estimates single station and remote reference electromagnetic transfer functions (TFs) from magnetotelluric (MT) time series.  Aurora is part of an open-source processing workflow that leverages the self-describing data container MTH5, which in turn leverages the general mt\_metadata framework to manage metadata.  These pre-existing packages simplify the processing workflow by providing managed data structures, transfer functions to be generated with only a few lines of code.  The processing depends on two inputs -- a table defining the data to use for TF estimation and a JSON file specifying the processing parameters, both of which are generated automatically and can be modified if desired.  Output TFs are returned as mt\_metadata objects, and can be exported to a variety of common formats for plotting, modeling, and inversion.  

## Key Features

- Tabular data indexing and management (Pandas dataframes), 
- Dictionary-like processing parameters configuration
- Programmatic or manual editing of inputs
- Largely automated workflow


# Introduction

Magnetotellurics (MT) is a geophysical technique for probing subsurface electrical conductivity using collocated electric and magnetic field measurements.  Field data is collected in the time domain, however the Earth can be approximated as a linear system in the frequency domain.  Therefore, common practice is to estimate the time invariant  (frequency domain) transfer function (TF) between electric and magnetic channels to get information of the Earth's resistivity structure [@egbert2002processing]. If measurements are orthogonal, the TF is equivalent to the electrical impedance tensor (Z) [@Vozoff:1991].  


$\begin{bmatrix} E_x \\ E_y \end{bmatrix}
=
\begin{bmatrix} 
Z_{xx} & Z_{xy} \\ 
Z_{yx} & Z_{yy} 
\end{bmatrix}
\begin{bmatrix} H_x \\ H_y \end{bmatrix}$
 
where ($E_x$, $E_y$), ($H_x$, $H_y$) denote orthogonal electric and magnetic fields respectively.  TF estimation requires the E and H time series _and_ metadata (locations, orientations, timestamps) along with a collection of signal processing and statistical techniques (@egbert1997robust and references therein).  The MTH5 data container archives metadata _with_ the data (@peacock2022mth5) and supplies time series as xarray (@hoyer2017xarray) objects for efficient, lazy access to data and easy application of scientific computing libraries available in Python.  

# Statement of Need

FORTRAN processing codes have long been available (e.g. EMTF @egbert2017mod3dmt, or BIRRP @chave1989birrp) but lack the readability of high-level languages and modifications to these programs are seldom attempted [@egbert2017mod3dmt], and have the additional barrier of compiling. Recently several Python versions of MT processing codes have been released by the open source community, including @shah2019resistics, @smai2020razorback, @ajithabh2023sigmt, and @mthotel.  Aurora adds to this canon of options but differs by leveraging the MTH5 and mt\_metadata packages eliminating a need for development of time series or metadata containers (@peacock2022mth5).  As a Python representation of Egbert's EMTF Remote Reference processing software, Aurora provides a continuity in the MT code space as the languages evolve.  Aurora is two degrees separated from the FORTRAN EMTF, as we used a Matlab implementation of EMTF from Prof. from Gary Egbert (Oregon State University, written communication, 2022-05-01) as an initial framework.  By providing an example workflow employing MTH5, we hope other developers may benefit from following this model, allowing researchers interested in signal-and-noise separation in MT to spend more time exploring and testing algorithms to improve TF estimates, and less time developing and redeveloping formats and management tools for data and metadata. Aurora is distributed under the [MIT](https://opensource.org/license/mit/) open-source license.


This manuscript describes the high-level concepts of the software – for information about MT data processing @ajithabh2023sigmt provides a concise summary, and more in-depth details can be found in @Vozoff:1991, @egbert2002processing and references therein.  

# Problem Approach

A TF instance depends on two key prior decisions: a) The data input to the TF computation algorithm, and b) The algorithm itself including the specific values of the processing parameters.  Aurora formalizes these concepts as classes (`KernelDataset` and `Processing`, respectively), and a third class `TransferFunctionKernel` (Figure \ref{TFK}), a composition of the `Processing`, and `KernelDataset`.  `TransferFunctionKernel` provides a place for validating consistency between selected data and processing parameters and specifies all information needed to make the calculation of a TF reproducible.

Generation of robust TFs can be done in only a few lines starting from an MTH5 archive.  Simplicity of workflow is due to the MTH5 data container already storing comprehensive metadata, including a channel summary table describing all time series stored in the archive including start/end times and sample rates.  Users can easily view a tabular summary of available data and select station pairs to process.  Once a station -- and optionally a remote reference station -- are defined, the simultaneous time intervals of data coverage at both stations are identified automatically, providing the `KernelDataset`.  Reasonable starting processing parameters are automatically generated for a given `KernelDataset`, and can be modified with code or via manual changes to a JSON file. Once the `TransferFunctionKernel` is defined, the processing automatically follows the flow described by Figure \ref{FLOW}.  Input time series are from a MTH5, these can initially be drawn from Phoenix, LEMI, FDSN, Metronix, Zonge, systems etc. and the resultant transfer functions can be exported to the most common TF formats such as .edi, .zmm, ,j, .avg, .xml etc.  The images of Figure  \ref{FLOW} are conceptual -- in reality the time series can have data from more than one station, and the spectrograms are also multivariate (not single channel as shown).  The regression is also multivariate, and applied on complex-valued data from the spectrograms, this illustration however conveys the key idea of regression in the presence of outliers and mixed clusters.

![TF Kernel concept diagram: Upper panel represents the TF Kernel with two inlay boxes representing the dataset (Pandas DataFrame) and a processing configuration (JSON). Lower panel illustrates example instances of these structures. Processing configuration image is clipped to show only a few lines.  \label{TFK}](TFKernelApproach.png)

![The main interfaces of Aurora TF processing. \label{FLOW}  Example time series from MTH5 archive in the linked notebook (using MTH5 built-in time series plotting), spectrogram from Fourier coeffcient (FC) data structure, regression cartoon from @hand2018statistical and TF from [SPUD](https://ds.iris.edu/spud/emtf/18633652).](aurora_workflow.png)

# Example

This section refers to a Jupyter notebook companion to this paper (archived on GitHub: [process_cas04_mulitple_station](https://github.com/simpeg/aurora/blob/joss/docs/tutorials/process_cas04_mulitple_station.ipynb)).  The companion notebook builds an MTH5 dataset from the EMscope dataset (@schultz2010emscope) and executes data processing -- a minimal_example is shown below.  Apparent resistivities are plotted in Figure \ref{compareTFs} along with the EMTF-generated results hosted at [EarthScope EMTF Spud](http://www.ds.iris.edu/spud/emtf/18633652).  


```python
from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.transfer_function.kernel_dataset import KernelDataset

run_summary = RunSummary()
run_summary.from_mth5s(["8P_CAS04_NVR08.h5",])
kernel_dataset = KernelDataset()
kernel_dataset.from_run_summary(run_summary, "CAS04", "NVR08")

cc = ConfigCreator()
config = cc.create_from_kernel_dataset(kernel_dataset) 

tf = process_mth5(config, kernel_dataset)
tf.write(fn="CAS04_rrNVR08.edi", file_type="edi")

```
<p id="minimal_example">Code snippet with steps to generate a TF from an MTH5. With MTH5 file ("8P_CAS04_NVR08.h5") in present working directory, a table of available contiguous blocks of multichannel time series is generated from `RunSummary()`.  In this example, the file contains data from two stations, "CAS04" and "NVR08" which are accessed from the EarthScope data archives.  Then station(s) to process are selected (by inspection of the `RunSummary` dataframe) to generate a `KernelDataset`.  The `KernelDataset` identifies simultaneous data at the local and reference site, and generates processing parameters, which can be edited before passing them to process_mth5, and finally export TF to a standard output format, in this case `edi`.</p>

To run the example you must install aurora, which can be done via conda or pip.  Detailed instructions and further documentation can be found on the SimPEG (@cockett2015simpeg) [documentation website](http://simpeg.xyz/aurora/).


![Comparison of apparent resistivities from Aurora and EMTF for station CAS04.  Both curves exhibit scatter in the low signal-to-noise ratio MT "dead band" between 1-10s, but most of estimates are very similar.  The Aurora results are from executing the example code snippet.  The plotting details are in the [Jupyter notebook](https://github.com/simpeg/aurora/blob/joss/docs/tutorials/process_cas04_mulitple_station.ipynb)(. \label{compareTFs}](tf_comparison.png)


# Testing
Aurora uses continuous integration [@duvall2007continuous] via unit and integrated tests, with ongoing improvement of test coverage.  Currently, CodeCov measures 77% code coverage (core dependencies mt_metadata and MTH5 at 84% and 60% respectively).  Aurora uses a small synthetic MT dataset for integrated tests.  On push to GitHub, the synthetic data are processed and the results compared against manually validated values (from Aurora and EMTF results) that are also stored in the repository.  Deviation from expected results causes test failures, alerting developers a code change resulted in an unexpected baseline processing result.  In the summer of 2023, wide-scale testing on EarthScope data archives was performed indicating that the Aurora TF results are similar to those form the EMTF Fortran codes, in this case for hundreds of real stations rather than a few synthetic ones.  Before release to common [PyPi](https://pypi.org/project/aurora/) and [Conda Forge](https://anaconda.org/conda-forge/aurora) repositories example Jupyter notebooks are also run via GitHub actions to assert functionality.




# Software Modifications
Aurora uses GitHub issues to track tasks and planned improvements.  We have recently added utilities for using a "Fourier coefficient" (FC) layer in the MTH5.  This allows for storage of the time series of Fourier coefficients in the MTH5, so the user can initialize TF processing from the FC layer, rather than the time series layer of the MTH5.  Prototype usage of this layer is already in Aurora's tests, but not part of the normal workflow.  Noise suppression techniques, for example coherence and polarization sorting and Mahalanobis distance (e.g. @ajithabh2023sigmt, @platz2019automated) could help reduce noise bias in the transfer functions.  A graphical data selection/rejection interface with time series plotting could allow users to manually weight data. The `TransferFunctionKernel` information could be implemented into both the MTH5 and the output EMTF\_XML (@kelbert2020emtf) for completeness. Unit and integrated tests could be expanded, including a test dataset from audio MT band. There are plans for Aurora to be co-developed with mt\_metadata, MTH5 and MTPy-v2 to maintain the ability to provide outputs for inversion and modeling. These improvements would support community participation in a comparative analysis of the open-source codes available to build a recipe book for handling noise from various open-archived datasets.


# Conclusion
Aurora provides an open-source Python implementation of the EMTF package for magnetotelluric data processing.  Processing is relatively simple and requires very limited domain knowledge in time series analysis. Aurora also serves as a prototype example of how to plug processing into an existing open data and metadata ecosystem (MTH5, mt_metadata, & MTpy-v2).  Aurora can be used as an example interface to these packages for the open source MT community, and these tools can contribute to workflows that allow more focus on geoscience analysis, and less on the nuances of data management.



# Acknowledgments 
The authors would like to thank IRIS (now EarthScope) for supporting the development of Aurora.  Joe Capriotti at SimPEG helped with online documentation and the initial release. Ben Murphy (Murphy Geo Consulting, LLC) provided methods for rotating impedance tensors from z-file formatted data. The facilities of the IRIS Consortium are supported by the National Science Foundation’s Seismological Facilities for the Advancement of Geoscience Award under Cooperative Support Agreement EAR-1851048. Any use of trade, firm, or product names is for descriptive purposes only and does not imply endorsement by the U.S. Government.


# References
```python

```