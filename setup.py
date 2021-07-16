import setuptools

setuptools.setup(
    name="aging-subtype-interface",
    version="0.1",
    author="Vikram Venkatraghavan",
    author_email="v.venkatraghavan@erasmusmc.nl",
    description="Backend functions for the webapplication",
    long_description="Same as above",
    long_description_content_type="text/markdown",
    url="https://github.com/subtypes-in-aging-brain/aging-subtype-interface",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
	'pandas',
	'numpy',
	'scikit-learn',
	'scipy',
	'matplotlib',
	'fpdf',
	'wand',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)