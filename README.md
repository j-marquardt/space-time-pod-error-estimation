# space-time-pod-error-estimation
## Description
This project provides the code for the paper "A-priori error estimation for space-time Galerkin POD for linear evolution problems" by Carmen Gräßle, Jan Heiland and Jannis Marquardt.

In this paper, we propose an a-priori error estimate for the model order reduction (MOR) method of space-time proper orthogonal decomposition (space-time POD). The original space-time POD approach extends standard POD by reducing not only the space dimension but simultaneously the time dimension as well. The proposed a-priori error estimate is developed for a linear parabolic partial differential equation and estimates the error between the numerical solution to a linear parabolic partial differential equation (PDE) and its space-time POD reduced solution. Numerical examples illustrate the occurring errors and analyze them in comparison to the theoretical bounds.

The code - as it is given in this repository - can be used to recreate the numerical examples from the paper. Furthermore, it can easily be altered for the implementation of own tests.

> [!NOTE]
> The code has not been optimized for efficiency. The implementation aims to be understandable and easily extendable instead of fast.

## System requirements
The code has been implemented and tested in `Python 3.9` in macOS 15.5. The following libraries/modules are required for the execution of the code:
- `sys`
- `matplotlib`
- `numpy`<br />
*(We remark that the `upper` keyword in `numpy.linalg.cholesky` requies a `numpy` version >= 2.0)*

## Usage
If you are new to `Python`, you have to [download and install](https://wiki.python.org/moin/BeginnersGuide) `Python`. The contents of this repository are available in GitHub under https://github.com/j-marquardt/space-time-pod-error-estimation.git.

The folder structure of this repository is as follows:
- `src` - Contains helper files which are reauired for the execution of the examples.
- `example_4_1` - Can be used to recreate Example 4.1 from the paper. 
- `example_4_2` - Can be used to recreate Example 4.2 from the paper.

> [!IMPORTANT]
> The examples can be executed independently from each other. They require the files from the `src` folder. If you want to rename the `src` folder or change the relative path from the executed file to the `src` folder, you also have to alter the corresponding lines in the `pod_error_comparison_????` files.

Each numerical test in the `example_??_!!` folder consists of two files:
- An executable file which starts with the prefix `pod_error_comparison_????`. You can execute these files from the command line without any additional arguments.
- A settings file in which the problem settings for the corresponding numerical test are specified. You can change the problem settings of a numerical tests in this file.


## License
This project is licensed under the terms of the ATTRIBUTION-SHAREALIKE 4.0 INTERNATIONAL ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en)) license.

## Contact
For questions, suggestions and bugs, please contact<br/>
Jannis Marquardt

Institute for Partial Differential Equations<br/>
Technische Universität Braunschweig<br/>
Universitätsplatz 2<br/>
38106 Braunschweig<br/>
Germany

Mail: j.marquardt@tu-braunschweig.de <br/>
Web: https://www.tu-braunschweig.de/en/ipde/jmarquardt

