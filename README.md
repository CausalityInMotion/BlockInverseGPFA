# GPFA_for_sklearn

This package is an implementation of the GPFA by Byron Yu et al. in python. \
Plus additional functional implementations 

## Usage

### Importing modules across subdirectories

To facilite importing modules across subdirectories, use the follow code. \
This is especially useful when runing the `../tutorial/gpfa.ipynb` file.
```
import pathlib
import sys
filepath = pathlib.Path.cwd()  # create path

# add path to python path
sys.path.insert(0, str(filepath.parent))
```
e.g., 
```
from gpfa import GPFA
```

### Tests

To run the unittests in the `test` folder, use the following command in your command line/terminal:
```
$ python -m unittest test.test_gpfa
```

## License
Modified BSD License based on Elephant, see [LICENSE.txt](LICENSE.txt) for details.


## Copyright

:copyright: 2021 Brooks M. Musangu and Jan Drugowitsch