# Connection-Analysis

README:

This program is used for analysing magnetic connection between interplanetary 
plasma shocks and Earth's bow shock.
Original Matlab code by Dr. Heli Hietala

Required data for succesful analysis:
-Magnetometric data
-Positional data of the satellite in GSE-coordinates
-Plasma data (= Speed of solar wind, density of plasma)


Note that the instructions provided here may not work for you if you are
not using Linux.

#===============================================================================

Instructions:

1) Acquiring data:

First you need to save the data needed for connection analysis to your computer. 
This can be done for example by using Nasa's Coordinated Data Analysis Web 
(CDAWeb). There you can chooce which spacecraft's data you are interested in, 
and then specify which data you would like to extract.

Magnetometric and plasma data can be collected by choosing 
"Magnetic Fields (space)" and "Plasma and Solar Wind". Specify the instruments
and time windows, and download the data in .txt format.

Positional data can be collected using Nasa's Satellite Situation Center Web
(SSCWeb). There you want to choose "Locator Tabular". Once there, choose the 
correct spacecraft. Input time windows so that they match with magnetometric
and plasma data, if possible. Failing to do so lowers the quality of the
analysis.
You'll need to specify output options. Do this by checking the first (X,Y,Z) box in 
GSE-coordinates. Do not check any other boxes, as those are
irrelevant to this analysis.
For output format, check that these conditions apply: Date=yy/mm/dd, 
time=hh:mm, Distance=Earth Radii, Degree's Format=-dd.ddd... to 2 decimals, 
Direction/Range=Lat(-90,+90),Long(0,360), Output Format=CDF. Save the .cdf file.

2) Preparing the program:

Now you should have three separate datafiles: positional data in .cdf
format, and magnetometric data and plasma data in .txt format. Make sure that
all these data files are in the same directory with connection_point.py.

Open connection_point.py and go to main(). There you'll find variables called
R_file, B_file and P_file. Set R_file to equal the name of your positional data.
Set B_file to equal the name of your magnetometric data. Set P_file to equal
the name of your plasma data.

Depending on the specific spacecraft in question, you might need to alter the
way this software transfers data into a dataframe. For this, you need to open
both your magnetometric data and your plasma data.
Make sure that the number of the first data line equals the "skiprows" variable
in their respective reading functions. The reading functions, both found in 
main() are called by the name of pd.read_csv(). Also make sure that all the 
columns in the datafiles are found from the list called "names". This list is
also found in the pd.read_csv() reading function.

DO NOT TOUCH ANYTHING NOT EXPLICITLY INSTRUCTED TO IN THIS PROGRAM TO AVOID
MALFUNCTION.

3) Running the program:

By this point you have three datafiles and connection_point.py in the same 
folder. You have set all the required variables in the main() -function to
correct values. 

If all conditions above apply, run the program in terminal by typing
"./connection_point.py" without parenthesis as an exact match.

The program will first notify you that the SpacePy version being used is not 
supported by the SpacePy team. You can ignore this message. The program will
then print out the size of the dataset and the current percentage of data 
analysed. 

When connection_point.py is finished, it will let you know by printing "Data 
analysed succesfully!". The program then proceeds to create a .txt file called 
"connection_data.txt", which contains the results of the analysis.

#===============================================================================

About the output of this program:

connection_data.txt will contain five columns as functions of time. These 
columns are f_min, r_bs, l_hit, Mms and theta. 

f_min contains the information on wether there is magnetic connection in
that specific moment of time or not. If f_min is nan, then there is no
connection. If it has a numerical value, there is connection

r_bs is the corresponding distance of the spacecraft from the bow shock, at the
moment of connection.

l_hit is the index number of r_bs in the program.

Mms is the magnetosonic mach number of the interplanetary shock.

theta is the obliquity of the interplanetary shock.

