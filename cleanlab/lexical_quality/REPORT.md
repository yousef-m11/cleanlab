## Running modified cleanlab fork by Yousef Mahmoud

This cleanlab was run and developed using Python 3.10.8 on macOS but should support other operating systems. All extra dependencies added in requirements-dev.txt. Should run as cleanlab would usually do - if not, uncomment final function call in lexical_filter.py to replicate the local testing I had conducted.

**Main additions**
1. lexical_quality folder contains lexical_filter.py - this contains functions which 
2. This was tested on a Stanford AI Library for movie reviews to ensure the lexical quality analysis is of a good standard - produced dataframe with the expected results.
3. filter.py modified to include call to lexical_filter.py so that it can then use the returned dataframe to determine which text and labels to exclude before ommencing the CL algorithm.

