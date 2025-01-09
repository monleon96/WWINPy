# WWPy - Weight Window Python Library

[![Version](https://img.shields.io/badge/version-0.1.3-blue.svg)](https://github.com/monleon96/WWPy)

WWPy is a Python library designed for working with Weight Window files (WWINP format). The library enables you to read, query, modify, and rewrite these files efficiently.

## Documentation

Full documentation is available at [Read the Docs](https://wwpy.readthedocs.io/).

## Installation

```bash
pip install wwpy
```

## Features

- Read WWINP format files
- Query weight window data
- Modify weight windows
- Write modified data back to WWINP format

## Quick Start

```python
import wwpy

# Read a weight window file
ww = wwpy.from_file("path/to/your/wwinp")

# Access weight window data
print(ww.header.ne)
print(ww.mesh.energy_mesh)
print(ww.values.ww_values)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details <https://www.gnu.org/licenses/>.
