# Accurate Estimation of Diffusion Coefficients and their Uncertainties from Computer Simulation

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/bjmorgan/kinisi/blob/master/docs/source/_static/schematic_light.png?raw=true">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/bjmorgan/kinisi/blob/master/docs/source/_static/schematic_dark.png?raw=true">
  <img alt="A schematic of the process to estimate the self-diffusion coefficient." src="https://github.com/bjmorgan/kinisi/blob/master/docs/source/_static/schematic_dark.png?raw=true">
</picture>

<p align="justify">
Self-diffusion coefficients, <i>D*</i>, are routinely estimated from molecular dynamics simulations by fitting a linear model to the observed mean-squared displacements (MSDs) of mobile species.
MSDs derived from simulation suffer from statistical noise, which introduces uncertainty in the resulting estimate of <i>D*</i>.
An optimal scheme for estimating <i>D*</i> will minimise this uncertainty, i.e., will have high statistical efficiency, and will give an accurate estimate of the uncertainty itself.
We present a scheme for estimating <i>D*</i> from a single simulation trajectory with high statistical efficiency and accurately estimating the uncertainty in the predicted value.
The statistical distribution of MSDs observable from a given simulation is modelled as a multivariate normal distribution using an analytical covariance matrix for an equivalent system of freely diffusing particles, which we parameterise from the available simulation data. 
We then perform Bayesian regression to sample the distribution of linear models that are compatible with this model multivariate normal distribution, to obtain a statistically efficient estimate of <i>D*</i> and an accurate estimate of the associated statistical uncertainty.
</p>

---

<p align="center">
<a href="https://arxiv.org/abs/2305.18244">
<img src="https://img.shields.io/badge/arXiv-2305.18244-orange.svg"/>
</a>
<a href="https://doi.org/10.1021/acs.jctc.4c01249">
<img src="https://img.shields.io/badge/publication%20DOI-10.1021/acs.jctc.4c01249-yellow.svg?style=flat"/>
</a>
<br><br>
<a href="https://orcid.org/0000-0003-3381-5911">Andrew R. McCluskey</a>&ast;, 
<a href="https://orcid.org/0000-0001-9722-5676">Samuel W. Coles</a> 
and 
<a href="https://orcid.org/0000-0002-3056-8233">Benjamin J. Morgan</a>&dagger;<br>
&ast;<a href="mailto:andrew.mccluskey@ess.eu">andrew.mccluskey@bristol.ac.uk</a>/&dagger;<a href="mailto:b.j.morgan@bath.ac.uk">b.j.morgan@bath.ac.uk</a>
</p>

---

This is the electronic supplementary information (ESI) associated with the publication "Accurate Estimation of Diffusion Coefficients and their Uncertainties from Computer Simulation". 
This ESI uses [`showyourwork`](https://show-your.work) to provide a completely reproducible and automated analysis, plotting, and paper generation workflow. 
To run the workflow and generate the paper locally using the cached data run the following: 
```
git clone git@github.com:arm61/msd-errors.git
cd msd-errors
pip install showyourwork
showyourwork build 
```
Full details of the workflow can be determined from the [`Snakefile`](https://github.com/arm61/msd-errors/blob/main/Snakefile) and the [`showyourwork.yml`](https://github.com/arm61/msd-errors/blob/main/showyourwork.yml).
