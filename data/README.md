## Testing Data

Usage:

```bash
cd data
cmake -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build
./data_generator <GENOMES_NUM> <THREADS_NUM>
```

Usage of data_generator:
```
Usage: ./data_generator <GENOMES_NUM> <THREADS_NUM>
<GENOMES_NUM> - number of genomes to generate
<THREADS_NUM> - number of threads to use for generation
```
