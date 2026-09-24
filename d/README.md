# D bindings for NumKong

Currently the SIMD libraries from C are not properly handled.

Also, the example assumes that the library will be downloaded from the GitHub Releases as a shared lib and saved to the folder of the example.

## Architecture
Current approach has configuration of the SIMD abilities in `numkong_c.c` file.
Feel free to switch off libraries that currently are not possible to read with ImportC.

Folder `c_header` is the same as `include` with the only difference:
```c
from_buffers[static 8]
```
changed with
```c
*from_buffers
```
as currently this C99 feature is not implemented in ImportC.

## Future improvements
When SIMD C libraries will be ported or will be supported into ImportC it will be possible to switch on them in `numkon_c.c`.
