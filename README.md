# VirtualFleet-recovery: Argo float recovery helper

The goal of this repository is to provide a library to make Argo floats trajectory predictions in order to facilitate recovery.
The prediction ``cone`` will be displayed on the map at: https://floatrecovery.euro-argo.eu/

### How it works ?
1. Given a specific float cycle, we extract:
   - space/time position of the last cycle, 
   - configuration parameters, such as parking depth, profiling depth and cycling period using the EA API.

2. We download the hourly CMEMS velocity fields for the region around the last float cycle

3. We run a VirtualFleet simulation where:
    - in addition to the past configuration, we define a list of modified configuration parameters, such as a reduced cycling frequency or parking depth
    - for each of these configurations, we set-up virtual floats with a random perturbations the float position in space/time

4. For each of the virtual float configurations, we compute a probabilistic `recovery` patch (basically the next profiles positions probability).