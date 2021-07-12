This is a research code base used to generate the ensemble in the multi-scale code base. An improved version with cleaner code and significantly more functionality will be released soon.  This preliminary version will be kept along with it's tag.

To run the code, navigate to src and run `runScript.py`

See `src/initializer.setRunParametersFromCommandLine` for commandline arguments and how to run the code under different parameters.

In the input files, the first column refers to the row of the VTD associated with the corresponding shapefile.

To run the chain with a different initial condition, modify line 70 of initializer.py to set the initPlan variable to be one of the following: "Plan16", "Plan20", "Rand0", "Rand1", "Rand2", ..., "Rand8"

The output files are placed in Output/NC and a new folder is created for different initial seeds, population constraints, and initial constraints.  The output plans can be found within the Samples directory in the generated output directory.  

There is a file for each output plan. The files are indexed by the step they were encountered at and only when a move was accepted.  For example, the directory structure

0.txt
1.txt
4.txt
...

Means that in the initial plan (found in 0.txt), there was an accepted proposal to move to the plan described in file 1.txt. This plan under went two rejected proposals before ending up with an accepted proposal that moved to the plan described in file 4.txt.

The output plans are reported in a multiscale framework. The first column holds a node description, the second the level (1 for county, 0 for precinct), and the third holds the district assignment.  For example

...
rockingham	1	11
1843	0	5
...

means that rockingham county belongs to district 11 and precinct 1843 belongs to district 5.

If one wishes to simulate elections on the output plans, we have provided four sets of election data that are located in inputData/NC

NC_PRESIDENT_12.txt
NC_PRESIDENT_16.txt
NC_USSENATE_14.txt
NC_USSENATE_16.txt

The first column olds the precinct description, the third column holds the number of Democratic votes in the given election, and the fourth holds the number of Republican votes (the last column is the number of independent or other votes and the second is a deprecated dummy column).

One may determine which county each precinct resides in via the file inputData/NC/NC_Counties.txt.

By combining (i) an output districting plan (e.g. Output/NC/.../Samples/4.txt), (ii) a precinct to vote count data (e.g. inputData/NC/NC_PRESIDENT_12.txt), and (iii) precinct to county information (e.g. inputData/NC/NC_Counties.txt), one can determine the number of Democrat and Republican votes in each district.

If one wanted to use this code base on another state, they would need to replicate the data structures in inputData/NC

Most of these data files are fairly straight forward: the first column holds the precinct identifier and the second a piece of data.  

One exception is NC_Edges_Lengths.txt is used to determine the graph adjacency and the first two columns hold adjacent precinct identifiers and the third holds the border length (which may be set to 1).  An identifier not found in the precinct identifiers (we use -1) is used to specify that a node is on the border of the state and takes place of a precinct identifier (i.e. is within the first two columns).

Another exception is the NC_MetaData.txt file which ties the different files together.  The first column holds the information as it will be read by the code base, the second the file, the ID:# represents the column on which to look up the precinct identifier and the col:# represents the column to look up the corresponding data.  

Node data contains one identifier and (at least) one piece of data, e.g. 

County	NC_Counties.txt	ID:0	col:1

Edge data contains two identifiers and on piece of data, e.g. 

BorderLength	NC_Edges_Lengths.txt	ID:0,1	col:2



Edge data contains 
