|<img src="https://raw.githubusercontent.com/euroargodev/VirtualFleet_recovery/master/docs/img/logo-virtual-fleet-recovery.png" alt="VirtualFleet-Recovery logo" width="400"><br>``Virtual Fleet - Recovery`` is a CLI to make predictions of Argo float positions|
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                                 [![DOI](https://zenodo.org/badge/543618989.svg)](https://zenodo.org/badge/latestdoi/543618989)                                                                                  |

The goal of this repository is to provide a CLI and Python library to make Argo floats trajectory predictions easy, in order to facilitate recovery.  

More about Argo floats recovery in here: 
- https://floatrecovery.euro-argo.eu  
- https://github.com/euroargodev/recovery/issues


# Documentation

## Command Line Interface

Primary groups of commands are ``predict`` and ``describe``.

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
Usage: vfrecovery describe [OPTIONS] WMO [CYC]...

  Returns data about an existing VirtualFleet-Recovery prediction

  Data could be a JSON file, specific metrics or images

Options:
  --log-level [DEBUG|INFO|WARN|ERROR|CRITICAL|QUIET]
                                  Set the details printed to console by the
                                  command (based on standard logging library).
                                  [default: INFO]
  -h, --help                      Show this message and exit.

  Examples:

  vfrecovery describe 6903091

  vfrecovery describe 6903091 112
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

## Making predictions

```bash
vfrecovery predict WMO CYC
vfrecovery predict WMO CYC1 CYC2 CYC3
```

Options:
```bash
vfrecovery predict --n_predictions 3 WMO CYC0
vfrecovery predict -np 3 WMO CYC0

vfrecovery predict --n_floats 2000 WMO CYC
vfrecovery predict -nf 2000 WMO CYC

vfrecovery predict --velocity GLORYS WMO CYC
vfrecovery predict -v GLORYS WMO CYC

vfrecovery predict --cfg_parking_depth 200 WMO CYC

vfrecovery predict --cfg_cycle_duration 60 WMO CYC

vfrecovery predict --cfg_profile_depth 1000 WMO CYC
```

## Describe results

```bash
vfrecovery describe velocity WMO CYC
vfrecovery describe obs WMO CYC1 CYC2 CYC3
```

## Other possible commands

```bash
vfrecovery meetwith "cruise_track.csv" WMO CYC0
```
