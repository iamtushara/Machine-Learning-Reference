# Welcome to NumPy tutorial

Use these Jupyter Notebooks to learn and understand how NumPy works. 
The Excersise folder contains questions for practice. Solutions are present in a different notebook in the same folder.


I'm often asked - "**What if I don't want NumPy to display values in Scientific Notation?**" and "**How to display multiple columns of a dataset in one row, without linebreaks in between**?"

Well, NumPy allows us to customize how values are displayed on the console. If you are having the same problem described above, Read the documentation below to find out how to deal with such situations.

# Setting print/display options in NumPy


**numpy.set_printoptions**(_`precision`=None, `threshold`=None, `edgeitems`=None, `linewidth`=None, `suppress`=None, `nanstr`=None, `infstr`=None, `formatter`=None, `sign`=None, `floatmode`=None, **kwarg_)



##### Parameters

`suppress` : bool, optional
>_If True, always print floating point numbers using fixed point notation, in which case numbers equal to zero in the current precision will print as zero. If False, then scientific notation is used when absolute value of the smallest number is < 1e-4 or the ratio of the maximum absolute value to the minimum is > 1e3. The default is False._

`linewidth` : int, optional
>_The number of characters per line for the purpose of inserting line breaks (default 75)._



##### Example
> numpy.set_printoptions(suppress=True, linewidth=150)

* Setting `supress=True` will prevent NumPy from displaying values in scientific notation
* Setting `linewidth=150` will let NumPy print at-a-maximum 150 characters in one line before giving linebreaks


##### Further Reading
https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
