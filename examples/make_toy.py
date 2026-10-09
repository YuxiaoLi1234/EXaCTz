"""Generate a deterministic finite 32^3 raw float64 smoke input; stdlib only."""
import array
import math
from pathlib import Path
values = array.array('d', (math.sin(x / 5) + math.cos(y / 7) + z / 31
                         for z in range(32) for y in range(32) for x in range(32)))
with Path('toy_f64.bin').open('wb') as output:
    values.tofile(output)
print('toy_f64.bin: 32 x 32 x 32 native-endian float64, 262144 bytes')
