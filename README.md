# Workshop-NeuralODE-2020
A 30-minute showcase on the how and the why of neural differential equations. 

## Content
There are three julia notebooks that show,

[Notebook 1:](./01-machine-learning.ipynb) an introduction to machine learning and neural networks,

![reaction rates](./figs/anns.png)

[Notebook 2:](./02-NODEs.ipynb) learning reaction kinetics using neural differential equations (NODEs),

![training](./figs/training.gif)

![reaction rates](./figs/reactionRate.png)

[Notebook 3:](./03-hybrid.ipynb) coupling these NODEs with reactor models.

## Installation instructions
### Installing Julia
1. Download the *Julia* binaries for your system [here](https://julialang.org/downloads/) we suggest to install the Long-term support release, v1.0.5
2. Yes, it is that simple :zap:

### Installing IJulia
Then run the Julia application
(double-click on it); a window with a `julia>` prompt will appear.  At
the prompt, type:
```julia
using Pkg
Pkg.add("IJulia")
```
to install IJulia.

This process installs a [kernel specification](https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernelspecs) that tells Jupyter (or JupyterLab) etcetera
how to launch Julia.

`Pkg.add("IJulia")` does not actually install Jupyter itself.
You can install Jupyter if you want, but it can also be installed
automatically when you run `IJulia.notebook()` below.  (You
can force it to use a specific `jupyter` installation by
setting `ENV["JUPYTER"]` to the path of the `jupyter` program
before `Pkg.add`, or before running `Pkg.build("IJulia")`;
your preference is remembered on subsequent updates.

### Installing the packages (first time running workshop)

After downloading Julia clone/download this repository to a location of your choice. Then start Julia. For *linux/MAC users* just use the command line and navigate to your local copy of the repository.
For *windows users* copy the path to the repository and navigate Julia to that directory as follows,
type ";" to go to the bash shell (red),

```bash
cd /change/this/to/the/location/of/Workshop-NeuralODE-2020 
```

Now that Julia has been navigated to the right working directory regardless of the operating system we can start the workshop.

In the normal Julia shell (green), activate the package and install the right packages. This can take a few minutes.

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()  # only the first time you run this environment
```

After a succesful installation of the packages the jupyter notebooks can be started.

### Running the notebook

type the following in Julia, at the `julia>` prompt:
```julia
import Pkg
Pkg.activate(".")
using IJulia
notebook()
```
to launch the IJulia notebook in your browser.

The first time you run `notebook()`, it will prompt you
for whether it should install Jupyter.  Hit enter to
have it use the [Conda.jl](https://github.com/Luthaf/Conda.jl)
package to install a minimal Python+Jupyter distribution (via
[Miniconda](http://conda.pydata.org/docs/install/quick.html)) that is
private to Julia (not in your `PATH`).
On Linux, it defaults to looking for `jupyter` in your `PATH` first,
and only asks to installs the Conda Jupyter if that fails; you can force
it to use Conda on Linux by setting `ENV["JUPYTER"]=""` during installation (see above).  (In a Debian or Ubuntu  GNU/Linux system, install the package `jupyter-client` to install the system `jupyter`.)

[source](https://raw.githubusercontent.com/JuliaLang/IJulia.jl/master/README.md)


-----
Contact: [Bram.De.Jaegher@gmail.com](mailto:bram.de.jaegher@gmail.com)

![footer](./figs/footerLogo.svg)
