#!/usr/bin/python3

'''
This program analyses potential magnetic connectivity of coronal mass ejection -driven
interplanetary plasma shocks and Earth's bow shock. 

Based on a MatLab implementation by Dr. Heli Hietala.

@Author: Christian Palmroos    <chospa@utu.fi>
Last updated: 23.01.2020

Potential names for this program:

Analyzer of Potential Magnetic connections between Interplanetary Plasma shocks and Earth's Bow Shock
APMIPEWS

Magnetic Connections Data Analysis
MaCoDA

Plasma Shock Connectivity Analysis
PSCA

Plasma Shock Analysis: Magnetic Connectivity
PSAMC
'''

import numpy as np
import numpy.matlib
import pandas as pd
from datetime import datetime
import os
import time
import warnings

#Living on the edge
warnings.filterwarnings("ignore")

#----------------------------------------------------------------------------------

def findtime(timelist,moment_of_time):
    
    '''
    This function seeks from a list of times the moment that is equal or the next
    greater than the given moment. It then returns the index of that moment in
    the given list.

    Input:  
    timelist = list of times (n list)         
    moment_of_time = moment of time we want to find (timestamp)    

    Output:     
    index = the first index corresponding to greater than or equal moment of time (integer)     
    '''

    #Initialization:
    #If timelist[i] >= moment_of_time is never satisfied, then return len(timelist), 
    #which is an impossible index.
    index = len(timelist)

    for i in range(index):
        if(timelist[i] >= moment_of_time):
            index = i
            break

    return index

#----------------------------------------------------------------------------------

def merka05_surface_eq(n,V,B):

    '''
    This function calculates the surface equation of a bow shock object.    

    Input:  
    n = plasma density (scalar)     
    V = speed of solar wind (scalar)    
    B = magnetic field modulus (scalar)     

    Output:     
    FAC = Scale factor for the coordinate system (scalar)    
    A   = A list of parameters defining the equation of the plane (10 list)     
    '''

    #Permeability of vacuum and the mass of the proton:
    mu_0 = 4*np.pi*10**(-7) #[Vs/Am]
    m_p = 1.672631*10**(-27) #[kg]

    #Calculating relevant upstream values:
    Babs = B
    Vabs = V
    D = n #D = solar wind proton number density [cm^(-3)]
    rho = n*10**(6) #[m^(-3)]


    #Alfven speed:
    VA = (Babs*10**(-9))/1000/(np.sqrt(mu_0*rho*m_p))# [km/s]
    XMA = Vabs/VA #XMA = Alfvenic Mach number

    #Merka05 model parameters
    #-------------------------
    #Scaling factor of the coordinate system:
    FAC = ((D/7.0)*(Vabs/457.5)**2)**(0.1666667)

    #Fitting parameters for Alfven Mach number dependance:
    B11 =0.0063
    B12 =-0.0098
    B31 =0.8351
    B32 =0.0102
    B41 =-0.02980
    B42 =0.0040
    B71 =16.39
    B72 =0.2087
    B73 =108.3
    B81 =-0.9241
    B82 =0.0721
    B101 =-444.0
    B102 =-2.935
    B103 =-1930.0

    #Fitting parameters of the shock surface:
    A1 = B11 + B12*XMA
    A2 = 1.0
    A3 = B31 + B32*XMA
    A4 = B41 + B42*XMA
    A5 = 0.0
    A6 = 0.0
    A7 = B71 + B72*XMA + B73/(XMA-1)**2
    A8 = B81 + B82*XMA
    A9 = 0.0
    A10 = B101 + B102*XMA + B103/(XMA-1)**2

    #Not optimal at the dawn flank (doesn't react enough), otherwise ok
    e = 1.5
    d1 = e*0.07
    d3 = e*0.05
    d4 = e*0.03
    d7 = e*0.8
    d8 = e*0.29
    d10 = e*40

    A = [A1,A2,A3,A4,A5,A6,A7,A8,A9,A10]

    return [FAC, A]

#----------------------------------------------------------------------------------

def merka05_value(n,V,B,Rgse):

    '''
    This function calculates the value of a point given in relation to the equation
    of the plane modeling bow shock. f=0 means that a point lies withing the plane.     

    Input:  
    n = density of plasma (scalar)  
    V = speed of solar wind (scalar)    
    B = magnetic field modulus (scalar)     
    Rgse = position vector (contravariant 3-vector)     

    Output:     
    f = value of a point in relation to the equation of Merka's bow shock model (scalar)    
    '''

    [FAC,A] = merka05_surface_eq(n,V,B)

    #Transform into a scaled GPE coordinate system (4 degree aberrated in GSE):
    phi = -4*(np.pi/180) # [rad]

    R = irf_newxyz(Rgse, [np.cos(phi), np.sin(phi), 0.], [-np.sin(phi), np.cos(phi), 0.], [0.,0.,1.])

    Xn = np.zeros([len(R)])
    Yn = np.zeros([len(R)])
    Zn = np.zeros([len(R)])

    for i in range(len(R)):
        Xn[i] = R[i,0]*FAC
        Yn[i] = R[i,1]*FAC
        Zn[i] = R[i,2]*FAC

    f = A[0]*Xn**2 + A[1]*Yn**2 + A[2]*Zn**2 + 2*A[3]*Xn*Yn + 2*A[4]*Yn*Zn + 2*A[5]*Xn*Zn + 2*A[6]*Xn + 2*A[7]*Yn + 2*A[8]*Zn + A[9]

    return f

#----------------------------------------------------------------------------------

def merka05_normal(n,V,B,Rgse):

    '''
    This function calculates the outward pointing unit vector perpendicular to 
    the surface of Merka's bow shock model at a given point.    

    Input:  
    n = density of plasma (scalar)   
    V = speed of solar wind (scalar)    
    B = magnetic field modulus (scalar)  
    Rgse = position vector in gse coordinates (contravariant 3-vector)  

    Output:     
    normal = normal vector of bow shock surface at Rgse    
    '''
    
    #First calculate the surface equation:
    [FAC,A] = merka05_surface_eq(n,V,B)

    #R must be an array in an array, so that the coordinate transform works properly.
    #This is because irf_newxyz expects a nx3 matrix as an input.
    R = np.array([Rgse])

    #Transform into a scaled GPE coordinate system (-4 degree aberrated in GSE):
    phi = -4*(np.pi/180) # [rad]
    R = irf_newxyz(R, [np.cos(phi), np.sin(phi), 0.], [-np.sin(phi), np.cos(phi), 0.], [0.,0.,1.])

    Xn = np.zeros([len(R)])
    Yn = np.zeros([len(R)])
    Zn = np.zeros([len(R)])

    for i in range(len(R)):
        Xn[i] = R[i,0]*FAC
        Yn[i] = R[i,1]*FAC
        Zn[i] = R[i,2]*FAC

    dfx = 2*A[0]*Xn + 2*A[3]*Yn + 2*A[6]
    dfy = 2*A[1]*Yn + 2*A[3]*Xn + 2*A[7]
    dfz = 2*A[2]*Zn

    l = np.sqrt(dfx**2 + dfy**2 + dfz**2)
    
    #Again normal must be a 1x3 array for the same reason as above:
    normal = np.zeros((1,3))

    normal[0,0] = dfx/l
    normal[0,1] = dfy/l
    normal[0,2] = dfz/l


    #Transform back into GSE coordinate system, by rotating 4 degrees back forward:
    phi = 4*(np.pi/180) # [rad]
    normal = irf_newxyz(normal, [np.cos(phi), np.sin(phi), 0.], [-np.sin(phi), np.cos(phi), 0.], [0.,0.,1.])

    return normal

#----------------------------------------------------------------------------------

def irf_newxyz(inp,x,y,z):

    '''
    This function performs a coordinate transform to a vector or a list of vectors
    with given basis vectors.       

    Input:      
    inp = (nx3) matrix. If inp has over 3 columns, then this function treats the first
    column as time, and the rest of the columns as a vector to be operated on.      
    x,y,z = basis vectors of the desired coordinate system (x=[xx,xy,xz], y=[yx,yy,yz], z=[zx,zy,zz])       
    
    Output:     
    out = Matrix with the same dimensions as inp.   
    '''

    #In case some of the vectors is a null vector, define it based on the other two
    if(x==0):
        x = np.cross(y,z)
        z = np.cross(x,y)
    if(y==0):
        y = np.cross(z,x)
        x = np.cross(y,z)
    if(z==0):
        z = np.cross(x,y)
        y = np.cross(z,x)

    #Make sure that x, y and z are unit vectors
    x = x/np.linalg.norm(x)
    y = y/np.linalg.norm(y)
    z = z/np.linalg.norm(z)

    #Initialize output:
    out = np.matrix(inp)

    x = np.matrix(x)
    y = np.matrix(y)
    z = np.matrix(z)

    #Check that inp and out are at least 3-dimensional 
    if(np.size(out[0]) == 3):

        for i in range(len(out)):
                out[i,0] = np.dot(inp[i],x.T)
                out[i,1] = np.dot(inp[i],y.T)
                out[i,2] = np.dot(inp[i],z.T)

    elif(np.size(out[0]) > 3):

        for i in range(len(out)):

            row = np.array([inp[i,1],inp[i,2],inp[i,3]])
            row = np.matrix(row)

            out[i,1] = np.dot(row,x.T)
            out[i,2] = np.dot(row,y.T)
            out[i,3] = np.dot(row,z.T)
    
    else:
        errStr="Coordinate transform impossible when input is less than 3 columns."
        raise Exception(errStr)

    return out

#----------------------------------------------------------------------------------

def connection(R,B,P):

    '''
    This function performs magnetic connectivity analysis on a given dataset. For each
    moment of time it calculates dynamically the equation of plane of Earth's bow shock
    according to Merka's bow shock model. It then checks if straight lines parallel to
    local magnetic field leaving the spacecraft hit this surface or not.    

    Input:      
    R  = positional dataframe. [unixtime, Rx, Ry, Rz] in GSE    
    B  = magnetometric dataframe. [unixtime,Bmag,Bx,By,Bz,B_bs] in GSE  
    P  = plasma dataframe.  [unixtime, V_bs, n_bs] in GSE   

    Output:     
    f_min    = min(f) of f. f=0 defines the bs plane in merka's model (scalar)        
    r_bs     = the coordinate in which we hit the bs (contravariant 3-vector)       
    l_hit    = index number of closest position to the bs (integer)         
    r_comp   = compression ratio (scalar)      
    theta_Bn = obliquity (scalar)          
    '''

    print("Magnetic connection analysis commencing... \n")

    #Initialize:
    tn = 100 #len(B['date_time'])
    l_hit = np.zeros((tn,1))
    f_min = np.zeros((tn,1))
    r_bs = np.zeros((tn,3))
    B_moments = np.zeros((tn,1))
    bs_normal = np.zeros((tn,3))

    print("Data set size:",tn)

    #R_epoch -> float
    #R['epoch'] = R['epoch'].astype(np.int64)
    #R['unix_time'] = R['epoch'].astype(np.int64)

    #Check the orientation of coordinate system:
    if(np.average(B['Bx']) > 0):
        B['Bx'] = -1*B['Bx']
        B['By'] = -1*B['By']
        B['Bz'] = -1*B['Bz']
    

    #Create the array "l" with 1800 elements.
    ln = 1800
    l = np.zeros(ln,dtype=int)
    for index in range(len(l)):
        l[index] = index+1
    l = np.matrix(l) #1x1800 matrix, [1,2,3,...,1799,1800]

    #Define unix_time vectors to be used in the upcoming loop:
    unxtim_B = B['unix_time']
    unxtim_B = np.array(unxtim_B)

    unxtim_R = R['unix_time']
    unxtim_R = np.array(unxtim_R)

    unxtim_P = P['unix_time']
    unxtim_P = np.array(unxtim_P)

    #If spacecraft is close to Earth, set first step to 0.1
    step_1 = 12
    if( np.mean(R['Xgse']) < 50 ):
        step_1 = 0.1

    #=============The connection analysis loop==============================================================
    for i in range(0,tn):

        #B_i is the i:th IMF vector from observed dataset
        B_i_row = B.iloc[i]
        B_i = np.array([B_i_row['Bx'],B_i_row['By'],B_i_row['Bz']])

        #The first index of R where time is greater than or equal to the 
        #moment of time at hand.
        try:

            i_r = findtime(unxtim_R,unxtim_B[i])

        except KeyError:

            print("KeyError at {}! No dataframe index found. Skipping to the next datapoint.".format(i))
            continue
        

        #If i_r is the same as the length of unxtim_R, then we have ran out of R and should use last r
        if(i_r == len(unxtim_R)):
            i_r = i_r - 1


        #r_sc is simply the coordinate vector of the sc.
        r_row = R.iloc[i_r]
        r_sc = np.array([r_row['Xgse'],r_row['Ygse'],r_row['Zgse']])

        #Convert B_i to matrix form, so python can do linear algebra with it.
        B_i = np.matrix(B_i) #1x3 matrix

        #r is a 1800x3 matrix, which contains 1800 coordinates that lie in the
        #straight line leaving the sc in the direction of B_i
        r = numpy.matlib.repmat(r_sc,ln,1) + step_1*(numpy.matlib.repmat(B_i,ln,1)) + 0.05*(np.dot(l.T,B_i))


        #========Calculate the dynamic bow shock==========
        
        #Find the moment of time from P corresponding to the moment at hand:
        try:
            i_p = findtime(unxtim_P,unxtim_B[i])
        except KeyError:
            print("KeyError at {}! No dataframe index found. Skipping to the next datapoint.".format(i))
            continue

        #If we run out of P, then BS cannot be generated, and should use the last value:
        if(i_p == len(unxtim_P)):
            i_p = i_p - 1

        P_row = P.iloc[i_p]
        n_bs  = P_row['n_bs']
        V_bs  = P_row['V_bs']
        B_bs  = P_row['B_bs']

        try:
            f = merka05_value(n_bs,V_bs,B_bs,r)
        except KeyError:
            continue #No connection

        #=================================================

        #Return the value that is closest to zero and the respective index
        l_closest = np.argmin(abs(f))
        f_min[i]  = abs(f[l_closest])

        #Set this position vector to be the position of the bow shock and
        #store the index in l_hit
        r_bs[i]  = r[l_closest]
        l_hit[i] = l_closest

        #Calculate the shock normal at r_bs: NOT WORKING AS OF NOW!!
        bs_normal[i] = merka05_normal(n_bs,V_bs,B_bs,r_bs[i])

        #Save the moments of time in B_moments:
        B_moments[i,0] = unxtim_B[i]

        #Print out the percentage of processed data to the terminal:
        percent = 100 * i/tn
        if(i % 2 == 0):
            print("Data processed: {} %".format(np.round(percent,decimals=2)), end='\r', flush=True)

    #==============Refresh terminal after exiting the main loop=============================================
    print(" ", end='\n',flush=True)


    #Once we exit the loop, cut the excess data from the endside away
    f_min = f_min[0:i+1]
    l_hit = l_hit[0:i+1]
    r_bs = r_bs[0:i+1,0:3]
    B_moments = B_moments[0:i+1]
    
    #... and update tn
    tn = len(f_min)

    
    #If f_min > 5, then we are too far from the bow shock to make connection.
    #These values are to be erased from the analysis.
    for i in range(tn):
        if(f_min[i] > 5):
            f_min[i] = np.nan
            r_bs[i] = np.nan
            l_hit[i] = np.nan


    #Initialize arrays for determining theta_Bn:
    Bx = np.array(B['Bx'])
    By = np.array(B['By'])
    Bz = np.array(B['Bz'])

    Bn = np.zeros(tn)
    theta_Bn = np.zeros(tn)


    #If B doesn't include Bmag, then make one
    try:

        Babs = B['Bmag']

    except KeyError:

        print("No Bmag included, constructing one from components")
        Babs = np.zeros(tn)
        for b in range(tn):
            Babs[b] = np.sqrt(Bx[b]**2 + By[b]**2 + Bz[b]**2)


    #Calculate theta_Bn:
    for k in range(tn):
        try:

            Bn[k] = bs_normal[k,0]*Bx[k] + bs_normal[k,1]*By[k] + bs_normal[k,2]*Bz[k]
            theta_Bn[k] = np.degrees(np.arccos(Bn[k]/Babs[k]))
            if( theta_Bn[k] > 90 ):
                theta_Bn[k] = 180 - theta_Bn[k]

        except KeyError:

            print("KeyError at {} when calculating theta_Bn! Expecting a float, got something else. Substituting NaN.".format(k))
            theta_Bn[k] = np.NaN
    

    '''
    Assuming radial flow of plasma, calculate the component parallel to the normal of
    BS in GSE coordinates.

    As of the present state of our research, we won't be needing this information. For
    this reason I will leave this part here commented.
    '''

    #V_radial = np.mean(P['V_bs'])
    #
    #Construct the tn x 4 solar wind matrix
    #V1 = np.ones((tn,1)) #"time"
    #V2 = V_radial*np.ones((tn,1)) #x-component
    #V3 = np.zeros((tn,1)) #y-component
    #V4 = np.copy(V3)
    #
    #V = np.array([[V1],[V2],[V3],[V4]])
    #V = np.matrix(V)
    #V = V.T
    #V_normal = np.zeros(tn)
    #
    #Dot product with velocity:
    #for i in range(tn):
    #    V_normal[i] = bs_normal[i,0]*V[i,1] + bs_normal[i,1]*V[i,2] + bs_normal[i,2]*V[i,3]
    


    '''
    Compression ratio for an oblique shock, set to 2 if x > -80 Re, otherwise 1.
    This is because in the current state of this program, we treat the compression ratio of
    the bow shock as a constant (which it's NOT). We know that the compression ratio of
    the BS decreases as we go further along the tail, until it is 1 (not a shock anymore).

    We define this radial limit for the length of the BS, so that we know which parts of space we
    are operating in.

    This should be varied between -80 and -100 in my opinion.
    '''

    bs_limit = -80 #in units of Earth radii

    r_comp = np.zeros(tn)
    for p in range(tn):
        if(r_bs[p,0] > bs_limit):
            r_comp[p] = 2.0
        else:
            r_comp[p] = 1.0
    

    #Reshape necessary lists so that np.hstack works on them
    r_comp = r_comp.reshape((tn,1))
    theta_Bn = theta_Bn.reshape((tn,1))


    #If r_comp == 1, then set other values to NaN
    for p in range(tn):
        if(r_comp[p] == 1.0):
            f_min[p] = np.NaN
            r_bs[p] = np.NaN
            l_hit[p] = np.NaN
            r_comp[p] = np.NaN
            theta_Bn[p] = np.NaN
        
    
    #Add time to all the lists to be returned
    f_min = np.hstack((B_moments,f_min))
    r_bs = np.hstack((B_moments,r_bs))
    l_hit = np.hstack((B_moments,l_hit))
    r_comp = np.hstack((B_moments,r_comp))
    theta_Bn = np.hstack((B_moments,theta_Bn))


    print(" ")
    print("Data analyzed succesfully!")

    return (f_min, r_bs, l_hit, r_comp, theta_Bn)

#----------------------------------------------------------------------------------

def convert_cdfs_to_dataframe(filelist, varlist, nameoftimecolumn, nameofvectorcolumn):

    '''
    import spacepy and delorean for cdfs and datetimes
    Remember (os.environ["CDF_LIB"] = "~/") before importing pycdf 
    '''

    os.environ["CDF_LIB"] = "~/"
    from spacepy import pycdf
    from delorean import Delorean

    #create empty numpy arrays
    ll=len(varlist); varsdata=[np.zeros(1) for i in range(ll+1)]
    #read data from cdf files and append the arrays.
    for i in filelist:
        d = pycdf.CDF(i)
        for j in varlist:
            idx=varlist.index(j)
            varsdata[idx]= np.append(varsdata[idx], pycdf.VarCopy(d[j]))

        
    #For create an epoch array from Epoch2
    #(s)econds (s)ince (epoch) == ssepoch
    idxe = varlist.index(nameoftimecolumn); ldata=len(varsdata[0]); ssepoch=np.zeros(ldata)
    vector1 = np.zeros(ldata-1)
    vector2 = np.zeros(ldata-1)
    vector3 = np.zeros(ldata-1)
        
    for i in range(1,ldata):
        ssepoch[i] = Delorean(varsdata[idxe][i],timezone="UTC").epoch 
        #drop the first zero before creating the data frame
        dictionary = {}; dictionary['epoch']=ssepoch[1:]
        for j in varlist:
            if j == nameoftimecolumn:
                dictionary['datetime']=varsdata[varlist.index(j)][1:]
            if j == nameofvectorcolumn:
                vector1[i-1] = varsdata[varlist.index(j)][1:][(i-1)*3]
                vector2[i-1] = varsdata[varlist.index(j)][1:][(i-1)*3+1]
                vector3[i-1] = varsdata[varlist.index(j)][1:][(i-1)*3+2]
                    
            else:
                dictionary[j] = varsdata[varlist.index(j)][1:]   
                    
    dictionary['vector1'] = vector1
    dictionary['vector2'] = vector2
    dictionary['vector3'] = vector3
    
    #Make the dataframe and replace all missing values with Nans
    d = pd.DataFrame(dictionary)
    d.replace(to_replace=-1e30,value=np.NaN,inplace=True)

    return d

#----------------------------------------------------------------------------------

def new_column_names(dataframe, column_names):

    '''
    This function changes the names of columns in a dataframe.  

    Input:  
    dataframe = the dataframe that has the columns you want to rename   
    column_names = a list of new column names   

    Output:     
    dataframe = new dataframe with the same data with renamed column names  
    '''

    cols = dataframe.columns.tolist()
    for index, values in enumerate(column_names):
            dataframe = dataframe.rename(columns={cols[index]:column_names[index]})
    
    return dataframe

#----------------------------------------------------------------------------------

def readcdfdata(filename):

    '''
    This function:      
    1) Reads a cdf datafile     
    2) Saves the data into a pandas dataframe       
    3) Replaces the dataframe's columns with new more descriptive names     
    4) Needs the functions "convert_cdfs_to_dataframe()" and "new_column_names()" to operate    

    =====================USER NOTICE======================================================  
    First print the contents of your cdf file. List those you want into your dataframe
    to "varlist". Then list the desired column names for your dataframe into "column_names".
    You might have to print the dataframe to see exactly how many columns it has, since
    a single cdf "var" may include multiple columns.
    '''

    filelist = [filename]
    varlist = ["Epoch","XYZ_GSE"]
    dataframe = convert_cdfs_to_dataframe(filelist, varlist,nameoftimecolumn="Epoch",nameofvectorcolumn="XYZ_GSE")
    column_names =["epoch","Datetime","Epoch","Xgse","Ygse","Zgse"]
    dataframe = new_column_names(dataframe,column_names)

    return dataframe

#----------------------------------------------------------------------------------

def printcdf(filename):

    '''
    Remember to use 'os.environ["CDF_LIB"] = library_directory' before import!

    For future notice: cdf_library = "cdf37_1-dist", and the best place for this library
    seems to be in "~/". pycdf apparently does not find this library from any
    other directories.
    '''

    os.environ["CDF_LIB"] = "~/"
    from spacepy import pycdf
    data = pycdf.CDF(filename)
    print(data)

#----------------------------------------------------------------------------------

def writedata(f,r,l,theta,sc_name):

    '''
    If one needs to save the analyzed data, use this function to generate a .txt
    file for that.      
    Notice! This function does not write down r_comp, since it's being treated 
    as a constant in the current version of this program.
    '''

    filename = "{}_connection_data.txt".format(sc_name)
    dataFile = open(filename, 'w')
    standard = True

    if(standard == True):

        dataFile.write("#time\t\t\tf_min\t\t\tr_bs\t\t\tl_hit\ttheta\n")
        for i in range(len(f)):
            dataFile.write(str(f[i,0]))
            dataFile.write("\t")
            dataFile.write(str(f[i,1]))
            dataFile.write("\t")
            dataFile.write(str(r[i,1]))
            dataFile.write("\t")
            dataFile.write(str(l[i,1]))
            dataFile.write("\t")
            dataFile.write(str(theta[i,1]))
            dataFile.write("\n")

    else:

        dataFile.write("#time\t\t\tf_min\t\t\tr_bs\t\t\tl_hit\ttheta\n")
        for i in range(len(f)):
            dataFile.write(str(f[0]))
            dataFile.write("\t")
            dataFile.write(str(f[i]))
            dataFile.write("\t")
            dataFile.write(str(r[i]))
            dataFile.write("\t")
            dataFile.write(str(l[i]))
            dataFile.write("\t")
            dataFile.write(str(theta[i]))
            dataFile.write("\n")
    
    print("Data saved into {}.".format(filename))

    dataFile.close()

#----------------------------------------------------------------------------------

def cutdata_at_shock(shock_crossing_time, dataframe):

    '''
    Cuts the excess data away onwards from the designated shock crossing time, and
    adds a unix time column to the end of the dataframe.         

    Input:  
    shock_crossing_time = time of shock crossing (str)      
    dataframe = the dataframe containing the data user wishes to cut    

    Output:     
    dataframe = the same dataframe that was given as input but now without unnecessary data
    '''

    shock_crossing_time = datetime.strptime(shock_crossing_time, "%Y-%m-%d %H:%M:%S")
    dates = dataframe['date_time']

    #If dataframe doesn't have unixtime, add it:
    if('unixtime' not in dataframe):
        t = pd.DatetimeIndex(dates, dayfirst=True)
        t = t.astype(np.int64)/(10**9)
        dataframe['unix_time'] = t


    #Then seek the row at which impact happens:
    for i in range(len(B)):

        lastrow = i

        try:
            if(datetime.timestamp(shock_crossing_time) < datetime.timestamp(dates[i])):
                break
        except KeyError:
            i = i-1
            continue


    #lastrow index has been found => update dataframe length
    dataframe = dataframe.iloc[:lastrow]

    return dataframe

#-----------------------------------------------------------------------------------

def offset_correction(dataframe, date_correction):

    '''
    This is a correction function needed for Geotail's magnetometric data.  

    Input:  
    dataframe = n x m dataframe    
    date_correction = n x 2 matrix. date1, correction1 etc.     

    Output:     
    dataframe with corrected Bz-component   
    '''

    r, c = date_correction.shape

    for i in range(c):
        
        date = datetime.strptime(date_correction[0,i], "%Y-%m-%d")
        offset = float(date_correction[1,i])

        for j in range(len(dataframe)):

            try:
        
                row = dataframe.iloc[j]

                if(datetime.date(row['date_time']) == datetime.date(date)):

                    row['Bz'] = row['Bz'] + offset
                    dataframe.iloc[j] = row
                
                elif(row['date_time'] > date):
                    break

            except KeyError:
                continue

    
    return dataframe

#-----------------------------------------------------------------------------------

if __name__ == "__main__":
    
    #Initialization:

    param_file_name = "analysis_params.txt"

    try:

        param_file = open(param_file_name,'r')

    except FileNotFoundError:

        errormsg = "Parameter file not found!"
        raise Exception(errormsg)

    #====Reading the params=================================================================================
    
    #R must be read in .cdf format. readcdfdata() handles it into a pd dataframe
    #B and P are probably easiest read as txt to pd dataframes.

    print("Reading parameters...")

    all_lines = param_file.readlines()
    params = np.zeros(8,dtype=object)

    line, param_index = 0, 0

    #Collect all lines which do not begin with '#' to params
    while(param_index < 8):

        param = all_lines[line]
        line = line + 1
        
        if(param[0] != '#'):
            params[param_index] = param
            param_index = param_index + 1


    #Handle all the parameters individually:
    #---------------------------------
    sc_name = str(params[0].split()[1]).capitalize()

    shock_crossing_time = str(params[1].split()[1]) + ' ' + str(params[1].split()[2])
    
    V_c = float(params[2].split()[1])
    
    cadence = int(params[3].split()[1])
    
    Rx = float(params[4].split()[1])

    R_file = str(params[5].split()[1])
    B_file = str(params[6].split()[1])
    P_file = str(params[7].split()[1])
    #---------------------------------

    #Close the file when finished with reading the parameters
    param_file.close()

    #=======================================================================================================

    
    #====POSITIONAL DATA IS READ FROM A .CDF FILE===========================================================
    
    print("Reading data...")

    R = readcdfdata(R_file)

    #R has 6 columns: date_time, unixtime, Epoch, Xgse, Ygse, Zgse
    R = R.rename(columns={'Datetime': 'date_time', 'epoch': 'unix_time'})
    
    #=======================================================================================================

    '''
    Determine dt = deltaX/Vc for each spacecraft to be used in calculating rolling mean. This has 
    to be done by hand, because rolling() understands row numbers, not time.

    For Vc we will use the average ambient upstream solar wind speed during this event Vc = 575 km/s
    For R[x] we use sc position at shock impact, therefore assuming it doesn't change much.

    For example:
    dt = R[x] / v_sw = 15*6370 km / 575 km/s = 166 s = 3 rows (Geotail)
    dt = R[x] / v_sw = 102.25*6370 km / 575 km/s = 1132.75 s = 378 rows (THEMISB)
    '''

    #====HERE WE READ MAGNETOMETRIC AND PLASMA DATA=========================================================

    if( sc_name == "Ace" ):

        #Ace's magnetometric data comes with 16 second cadence

        B = pd.read_csv(B_file, skiprows=62, skipfooter=4, parse_dates=[['date','time']], names=['date','time','Bmag','Bx','By','Bz'],delimiter=r"\s+",dayfirst=True,engine='python')
    
        dt = 165 #dt is not in seconds, but in rows.

    #----------------------------------------------------------------------------------------------------

    elif( sc_name == "Wind" ):

        #Wind's magnetometric data comes with 60 second cadence

        B = pd.read_csv(B_file, skiprows=105, skipfooter=4, parse_dates=[['date','time']], names=['date','time','Bmag','Bx','By','Bz'],delimiter=r"\s+",dayfirst=True,engine='python')

        dt = 40

    #----------------------------------------------------------------------------------------------------

    elif( sc_name == "Geotail" ):

        #Geotail's magnetometric data comes with 60 second cadence

        B = pd.read_csv(B_file, skiprows=42, skipfooter=4, parse_dates=[['date','time']], names=['date','time','Bmag','Bx','By','Bz'],delimiter=r"\s+",dayfirst=True,engine='python')

        #Modification due to geotail's magnetometric data's format:
        B['Bmag'] = 0.1*B['Bmag']
        B['Bx'] = 0.1*B['Bx']
        B['By'] = 0.1*B['By']
        B['Bz'] = 0.1*B['Bz']

        #Geotail's magnetometric data has an offset in the z-component. These are the correct offsets
        #for these dates.
        date_correction = np.array([["2010-04-04","2010-04-05","2010-04-06"],[0.22,0.21,0.20]])

        B = offset_correction(B,date_correction)

        dt = 3

    #----------------------------------------------------------------------------------------------------

    elif( sc_name == "Themisb" ):

        #THEMISB's magnetometric data comes with 3 second cadence
    
        B = pd.read_csv(B_file, skiprows=64, skipfooter=4, parse_dates=[['date','time']], names=['date','time','Bmag','Bx','By','Bz'],delimiter=r"\s+",dayfirst=True,engine='python')    

        dt = 378

    #----------------------------------------------------------------------------------------------------

    else:

        errormsg = "Spacecraft not recognized!"
        raise ValueError(errormsg)

    #----------------------------------------------------------------------------------------------------

    #Finally, no matter what spacecraft, we use OMNI data for solar wind and plasma values:
    P = pd.read_csv(P_file, skiprows=90, skipfooter=4, parse_dates=[['date','time']], names=['date','time','B_bs','Vx','Vy','Vz','n_bs'],delimiter=r"\s+",dayfirst=True,engine='python')

    '''
    ===============================USER NOTICE=========================================================
    OMNI dataset has been calculated so that the values at each moment of time correspond to values not
    at the position of the spacecraft, but at the nose of the bow shock. For this reason we don't need
    to do the transfer of values ourselves.

    Be mindful of what plasma dataset you use, for you may need to take the locality of observed 
    values in to account manually.
    '''

    #All data has been read by this point
    print("Success!")

    #=======================================================================================================


    #====WE SHOULD CUT THE DATA AT THE MOMENT THE SC HITS THE SHOCK=========================================

    R = cutdata_at_shock(shock_crossing_time,R)
    B = cutdata_at_shock(shock_crossing_time,B)
    P = cutdata_at_shock(shock_crossing_time,P)

    #Remember that cutdata also adds 'unix_time' -column to the dataframes!

    #=======================================================================================================


    #====AVERAGING WINDOW DEPENDS ON THE POSITION OF THE SC AND THE SPEED OF THE SOLAR WIND=================

    #Before starting averaging values, erase bad data values:
    B = B.loc[B['Bmag'] >= 0]
    P = P.loc[P['Vx'] <= 0]


    #Calculate the rolling mean values for B with pandas:
    B['Bmag'] = B.Bmag.rolling(window=dt, min_periods=1).mean()
    B['Bx']   = B.Bx.rolling(window=dt, min_periods=1).mean()
    B['By']   = B.By.rolling(window=dt, min_periods=1).mean()
    B['Bz']   = B.Bz.rolling(window=dt, min_periods=1).mean()

    #=======================================================================================================

    #Before starting the analysis, add V_bs to P:
    P['V_bs'] = np.sqrt(P['Vx']**2 + P['Vy']**2 + P['Vz']**2)

    #And delete unnecessary columns from dataframes:
    R = R.drop(['Epoch'], axis=1)
    P = P.drop(['Vx','Vy','Vz'], axis=1)

    #Run the connection analysis on provided data:
    f_min, r_bs, l_hit, r_comp, theta_Bn = connection(R,B,P)

    #After completing the analysis, write the data in a txt file:
    writedata(f_min,r_bs,l_hit,theta_Bn,sc_name)

    #This concludes the raw connection analysis.

    #=======================================================================================================
