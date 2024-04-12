|<img src="https://raw.githubusercontent.com/euroargodev/VirtualFleet_recovery/master/docs/img/logo-virtual-fleet-recovery.png" alt="VirtualFleet-Recovery logo" width="400"><br>``Virtual Fleet - Recovery`` is a CLI to make predictions of Argo float positions|
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                                 [![DOI](https://zenodo.org/badge/543618989.svg)](https://zenodo.org/badge/latestdoi/543618989)                                                                                  |

The goal of this repository is to provide a CLI and Python library to make Argo floats trajectory predictions easy, in order to facilitate recovery.  

More about Argo floats recovery in here: 
- https://floatrecovery.euro-argo.eu  
- https://github.com/euroargodev/recovery/issues


# Documentation

## Command Line Interface

Primary groups of commands are ``predict``, ``describe`` and ``db``.

### vfrecovery predict
```
Usage: vfrecovery predict [OPTIONS] WMO CYC

  Execute the VirtualFleet-Recovery predictor

  WMO is the float World Meteorological Organisation number.

  CYC is the cycle number location to predict. If you want to simulate more
  than 1 cycle, use the `n_predictions` option (see below).

Options:
  -v, --velocity TEXT             Velocity field to use. Possible values are:
                                  'GLORYS', 'ARMOR3D'  [default: GLORYS]
  --output_path TEXT              Simulation data output folder [default:
                                  './vfrecovery_simulations_data/<WMO>/<CYC>']
  --cfg_parking_depth FLOAT       Virtual floats parking depth in db [default:
                                  previous cycle value]
  --cfg_cycle_duration FLOAT      Virtual floats cycle duration in hours
                                  [default: previous cycle value]
  --cfg_profile_depth FLOAT       Virtual floats profile depth in db [default:
                                  previous cycle value]
  --cfg_free_surface_drift INTEGER
                                  Virtual cycle number to start free surface
                                  drift, inclusive  [default: 9999]
  -np, --n_predictions INTEGER    Number of profiles to predict after cycle
                                  specified with argument 'CYC'  [default: 0]
  -nf, --n_floats INTEGER         Swarm size, i.e. the number of virtual
                                  floats simulated to make predictions
                                  [default: 100]
  -s, --domain_min_size FLOAT     Minimal size (deg) of the simulation domain
                                  around the initial float position  [default:
                                  5]
  --overwrite                     Should past simulation data be overwritten
                                  or not, for a similar set of arguments
  --lazy / --no-lazy              Load velocity data in lazy mode (not saved
                                  on file).  [default: lazy]
  --log_level [DEBUG|INFO|WARN|ERROR|CRITICAL|QUIET]
                                  Set the details printed to console by the
                                  command (based on standard logging library).
                                  [default: INFO]
  -h, --help                      Show this message and exit.

  Examples:

          vfrecovery predict 6903091 112
 ```

### vfrecovery describe

```
Usage: vfrecovery describe [OPTIONS] TARGET WMO [CYC]...

  TARGET select what is to be described. A string in: ['obs', 'velocity',
  'run'].

  WMO is the float World Meteorological Organisation number

  CYC is the cycle number location to restrict description to

Options:
  --log-level [DEBUG|INFO|WARN|ERROR|CRITICAL|QUIET]
                                  Set the details printed to console by the
                                  command (based on standard logging library).
                                  [default: INFO]
  -h, --help                      Show this message and exit.

  Examples:

  vfrecovery describe velocity 6903091

  vfrecovery describe obs 6903091 112
 ```

### vfrecovery db

```
Usage: vfrecovery db [OPTIONS] ACTION

  Internal simulation database helper
  
Options:
  --log-level [DEBUG|INFO|WARN|ERROR|CRITICAL|QUIET]
                                  Set the details printed to console by the
                                  command (based on standard logging library).
                                  [default: INFO]
  -i, --index INTEGER             Record index to work with
  -h, --help                      Show this message and exit.

  Examples:

  vfrecovery db info

  vfrecovery db read

  vfrecovery db read --index 3

  vfrecovery db drop
```


## Python interface


### vfrecovery.predict

```python
import vfrecovery

wmo, cyc = 6903091, 126
results = vfrecovery.predict(wmo, cyc)
```

Signature:
```
vfrecovery.predict(
    wmo: int,
    cyc: int,
    velocity: str = 'GLORYS',
    output_path: Union[str, pathlib.Path] = None,
    n_predictions: int = 0,
    cfg_parking_depth: float = None,
    cfg_cycle_duration: float = None,
    cfg_profile_depth: float = None,
    cfg_free_surface_drift: int = 9999,
    n_floats: int = 100,
    domain_min_size: float = 5.0,
    overwrite: bool = False,
    lazy: bool = True,
    log_level: str = 'INFO',
)
```



# API Design

## Other possible commands

```bash
vfrecovery meetwith "cruise_track.csv" WMO CYC0
```

## Data storage
Simulation data are stored on disk under the following architecture:

```
./vfrecovery_simulations_data
                            |- vfrecovery_simulations.log
                            |- WMO
                               |----CYC
                                    |----VELOCITY(NAME + DOWNLOAD_DATE + DOMAIN_SIZE)
                                         |- velocity_file.nc
                                         |- figure.png
                                         |---- RUN_PARAMS(NP + CFG + NF)
                                               |- float_configuration.json
                                               |- trajectories.zarr
                                               |- results.json
                                               |- figure.png
```

This ensures that for a given velocity field, all possible simulations are unambiguously found under a single folder