#! /usr/bin/env python
# $Id: setup.py 1494 2009-04-30 13:10:41Z css1hs $

from distutils.core import setup, Extension

setup(
  ext_modules = [
    Extension ( "jpegObject",
      sources=["jpegobject.c"],
      library_dirs=[ "jpeglib" ],
      include_dirs=["/usr/include/python2.7",
                    "/usr/lib64/python2.7/site-packages/numpy/core/include/",
                ],
      libraries=[ "jpeg" ],
    ),
  ],
)

# Note that the include_dirs given here are due to a bug in Ubuntu.
# -> Check it for your system.
