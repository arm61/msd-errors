# Accurate estimation of diffusion coefficients and their uncertainties from computer simulation

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/bjmorgan/kinisi/blob/master/docs/source/_static/schematic_light.png?raw=true">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/bjmorgan/kinisi/blob/master/docs/source/_static/schematic_dark.png?raw=true">
  <img alt="A schematic of the process to estimate the self-diffusion coefficient." src="https://github.com/bjmorgan/kinisi/blob/master/docs/source/_static/schematic_dark.png?raw=true">
</picture>

<p align="justify">
Self-diffusion coefficients, <i>D*</i>, are routinely estimated from molecular dynamics simulation data by fitting a linear model to the observed mean-squared displacements (MSDs) of mobile species.
Molecular dynamics simulations are stochastic, and simulation MSDs suffer from statistical noise, which introduces uncertainty in the resulting estimate of <i>D*</i>.
An optimal scheme for estimating <i>D*</i> will minimise this uncertainty, i.e., will have high statistical efficiency, while also giving an accurate estimate of the uncertainty itself.
We present a scheme for estimating <i>D*</i> from a single simulation trajectory with high statistical efficiency while also accurately estimating the uncertainty in the predicted value.
The statistical distribution of MSDs observed from a simulation is modelled as a multivariate normal distribution, using an analytical covariance matrix derived for an equivalent system of freely diffusing particles.
We parameterise this covariance matrix using estimated variances for the observed MSD, obtained via rescaling of the simulation data variance. 
Sampling this model multivariate normal distribution using Bayesian methods gives a statistically efficient estimate of <i>D*</i> and an accurate estimate of the associated statistical uncertainty. 
</p>

---

<p align="center">
<a href="https://github.com/arm61/msd-errors/actions/workflows/build.yml">
<img src="https://github.com/arm61/msd-errors/actions/workflows/build.yml/badge.svg" alt="Article status"/>
</a>
<a href="https://github.com/arm61/msd-errors/raw/main-pdf/arxiv.tar.gz">
<img src="https://img.shields.io/badge/article-tarball-blue.svg?style=flat" alt="Article tarball"/>
</a>
<a href="https://github.com/arm61/msd-errors/raw/main-pdf/ms.pdf">
<img src="https://img.shields.io/badge/article-pdf-blue.svg?style=flat" alt="Read the article"/>
</a>
<a href="https://doi.org/10.5281/zenodo.xxxxxxx">
<img src="https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg"/>
</a>
<a href="https://arxiv.org/abs/xxxx.xxxxx">
<img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-orange.svg"/>
</a>
<br><br>
<a href="https://orcid.org/0000-0003-3381-5911">Andrew R. McCluskey</a>&ast;, 
<a href="https://orcid.org/0000-0001-9722-5676">Samuel W. Coles</a> 
and 
<a href="https://orcid.org/0000-0002-3056-8233">Benjamin J. Morgan</a>&dagger;<br>
&ast;<a href="mailto:andrew.mccluskey@ess.eu">andrew.mccluskey@ess.eu</a>/&dagger;<a href="mailto:b.j.morgan@bath.ac.uk">b.j.morgan@bath.ac.uk</a>
</p>

---

This is the electronic supplementary information (ESI) associated with the publication "Estimation of diffusive properties for *in-silico* materials using a Gaussian process". 
This ESI uses [`showyourwork`](https://show-your.work) to provide a completely reproducible and automated analysis, plotting, and paper generation workflow. 
To run the workflow and generate the paper locally using the cached data run the following: 
```
git clone git@github.com:arm61/msd-errors.git
cd msd-errors
pip install showyourwork
showyourwork build 
```
Full details of the workflow can be determined from the [`Snakefile`](https://github.com/arm61/msd-errors/blob/main/Snakefile) and the [`showyourwork.yml`](https://github.com/arm61/msd-errors/blob/main/showyourwork.yml).
