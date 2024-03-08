|<img src="https://raw.githubusercontent.com/euroargodev/VirtualFleet_recovery/master/docs/img/logo-virtual-fleet-recovery.png" alt="VirtualFleet-Recovery logo" width="400"><br>``Virtual Fleet - Recovery`` is a CLI to make predictions of Argo float positions|
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                                 [![DOI](https://zenodo.org/badge/543618989.svg)](https://zenodo.org/badge/latestdoi/543618989)                                                                                  |

The goal of this repository is to provide a CLI and Python library to make Argo floats trajectory predictions easy, in order to facilitate recovery.  
The library produces a prediction _patch_ or _cone_ that could be displayed on a map like here: https://floatrecovery.euro-argo.eu  
More about Argo floats recovery in here: https://github.com/euroargodev/recovery/issues

# Documentation

## Command Line Interface

Main commands:
```bash
vfrecovery predict WMO CYC
vfrecovery predict WMO CYC1 CYC2 CYC3

vfrecovery describe WMO CYC
vfrecovery describe WMO CYC1 CYC2 CYC3

vfrecovery whiterun WMO CYC
vfrecovery whiterun WMO CYC1 CYC2 CYC3

vfrecovery meetwith "cruise_track.csv" WMO CYC0
```

Options:
```bash
vfrecovery predict --n_predictions 3 WMO CYC0
vfrecovery predict -n 3 WMO CYC0

vfrecovery predict --n_floats 2000 WMO CYC
vfrecovery predict -nf 2000 WMO CYC

vfrecovery predict --velocity GLORYS WMO CYC
vfrecovery predict -v GLORYS WMO CYC

vfrecovery predict --quiet WMO CYC
vfrecovery predict -q WMO CYC

vfrecovery predict --cfg_parking_depth 200 WMO CYC
vfrecovery predict -cfg_pdpt 200 WMO CYC

vfrecovery predict --cfg_cycle_duration 60 WMO CYC
vfrecovery predict -cfg_clen 60 WMO CYC

vfrecovery predict --cfg_profile_depth 1000 WMO CYC
vfrecovery predict -cfg_pfdpt 1000 WMO CYC
```

```bash
vfrecovery predict --cfg_parking_depth [200, 1000] WMO CYC1 CYC2
```

## Python interface

```python
import vfrecovery

vfrecovery.predict(
    wmo=<WMO>,
    cyc=<CYCLE_NUMBER>,
    [OPTION] = <value>,
)
```