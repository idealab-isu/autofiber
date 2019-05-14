# AutoFiber
Strain energy minimization of geodesic based parameterizations over 3D
surfaces for optimization of fiber layup orientations. Created to be part
of the [De-La-Mo](http://thermal.cnde.iastate.edu/de-la-mo.xhtml)
automatic defect insertion into FEM package developed at Iowa
State University.

## Project Structure
* `autofiber/`: Contains the python package *autofiber*
* `demos/`: Contains a variety of demo models and a script, *test.py*
which demonstrates usage on each model.
* `doc/`: Contains various documentation materials

## Dependencies
* Requires Python 2.7 (limited by spatialnde)
* `spatialnde`: 3D model loader and image projection package \
Created by Dr. Stephen D. Holland at Iowa State University \
[Spatialnde](http://thermal.cnde.iastate.edu/spatialnde)
* `Numpy`
* `Matplotlib` - optional for plotting

## Installation
Once all dependencies are installed run:
```
python setup.py install
```

## How to run
Take a look at `demos/test.py` for an in-depth explaination of the relevant
API calls and how they work for a variety of models.

## Abaqus integration
A function has been created and implemented in De-La-Mo to allow for
automatic insertion of computed fiber orientations into Abaqus. See the
[De-La-Mo](http://thermal.cnde.iastate.edu/de-la-mo.xhtml) project for
more information.
