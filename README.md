# Lightning Queen

This is a CUDA project that is designed to compare different approaches for reading data from SSD to GPU for future analyzing of that data.
For testing data I chose to work with genomes and all the results wil be shown for short fake 100 MB long genomes.

## Testing Data

For generating fake 100 MB long genomes there is small project written in `data` directory.

Usage:

```bash
cd data_generator
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
