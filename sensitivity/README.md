## Known Issues

### `cantera_functions.py`

- For "bad" operating conditions (obtained from random sampling), IDT doesn't occur, so those points are skipped. However, sometimes this results in skipping too many points, which likely reduces accuracy. Needs to be investigated and handled better — possibly by rejecting bad operating conditions at the very beginning.
- Didn't have time to implement pressure rise correction. It is relevant for multi-species/mixed fuels. It wouldn't affect sensitivity too much, maybe just the rankings, but it should still be implemented.
- Didn't have time to implement handling of multiple local maxima.
- Implemented IDT for different targets and types, but a simple dT/dt maximum seems to work better. This likely means something is wrong with the current implementation. Needs to be thoroughly tested.
- Currently, any multiplicative factor is applied to the Cantera `net_rate`. It may be more useful and powerful to modify A, n, Ea directly, but that would need to be implemented.