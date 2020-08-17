This is an analysis of walking data done using pandas, and scikit-learn.
2 contributers collected walking data by strapping a phone to their ankles, holding it in their hands, and putting in their pockets.
Our goal is to use different Machine Learning techniques to be able to classify the person, as well as the position of the phone


This program requires a few dependencies:
```
numpy
pandas
matplotlib
scikit-learn
```

To install these dependencies run the following lines first:
```
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
```

This program comes with sample input files in the data directory

To run the program, use the following command line:
```
cd src
python3 project.py
```

Once the program completes, it will produce 4 png images: The post-hoc analysis for placement, person, Vafa(side), and Song(side). These are located in the src folder. 

The etc folder contains the graphs for the filtered and unfiltered x-axis data, as found in the report. It also contains the 2 summary csv's used within the program. Finally, there are 2 example outputs, one with velocity & position added, and one without. 
