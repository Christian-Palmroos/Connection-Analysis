#!/usr/bin/python3

'''
This program is used for analysis on when a spacecraft lies on magnetic fields lines that
connect it with Earth's bow shock.
Original MatLab code by Dr. Heli Hietala.

@Author: Christian Palmroos    <chospa@utu.fi>
Last updated: 11.10.2019
'''

import numpy as np
import numpy.matlib
import pandas as pd
from datetime import datetime
import os
import time

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

def h_findtime(x,t):

    #If x[i] >= t is never satisfied, then return len(x), which is an impossible index.
    u = len(x)

    for i in range(len(x)):
        if(x[i] >= t):
            u = i
            break

    return u

#----------------------------------------------------------------------------------

def merka05_surface_eq(n,V,B):
    '''
    D = solar wind proton number density [cm^(-3)]
    V = solar wind speed [km/s]
    XMA = Alfvenic Mach number
    '''
    mu_0 = 4*np.pi*10**(-7) #[Vs/Am]
    m_p = 1.672631*10**(-27) #[kg]

    #Calculating relevant upstream values:
    Babs = B
    Vabs = V
    D = n
    rho = n*10**(6) #[m^(-3)]

    #Alfven speed:
    VA = (Babs*10**(-9))/1000/(np.sqrt(mu_0*rho*m_p))# [km/s]
    XMA = Vabs/VA

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

    dfx = 2*A[0]*Xn + 2*A[3]*Yn + 2*A[6]
    dfy = 2*A[1]*Yn + 2*A[3]*Xn + 2*A[7]
    dfz = 2*A[2]*Zn

    l = np.sqrt(dfx**2 + dfy**2 + dfz**2)

    #Construct the R x 3 matrix that has all the normal vectors
    normal = np.matrix(R)
    for j in range(len(normal)):
        normal[j,0] = dfx[j]/l[j]
        normal[j,1] = dfy[j]/l[j]
        normal[j,2] = dfz[j]/l[j]

    #Transform back into GSE coordinate system, by rotating 4 degrees back forward:
    phi = 4*(np.pi/180) # [rad]
    normal = irf_newxyz(normal, [np.cos(phi), np.sin(phi), 0.], [-np.sin(phi), np.cos(phi), 0.], [0.,0.,1.])

    return normal

#----------------------------------------------------------------------------------

def irf_newxyz(inp,x,y,z):
    '''
    Input:
    inp = 3column vector. If inp has over 3 columns, then this function treats the first
    column as time, second, third and fourth as the columns to be operated on.
    x,y,z = coordinate vectors (x=[xx,xy,xz], y=[yx,yy,yz], z=[zx,zy,zz])
    
    Output:
    out = Matrix with the same dimensions as inp.
    '''

    #In case some of the vectors is a 0, define it based on the other two
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

#----------------------------------------------------------------------------------

def connection(R,B,nc,Vc,Bc):
    '''
    Input:
    R = [unixtime, Rx, Ry, Rz] in GSE
    B = magnetic field vector, B = [unixtime,Bx,By,Bz] in GSE
    nc = (c = constant) density
    Vc = Solar wind
    Bc = constant magnetic field strength

    Output:
    f_min= min(f) of f. f=0 defines the bs plane in merka's model
    r_bs= the coordinate in which we hit the bs
    l_hit= index number of closest position to the bs
    Mms= Magnetosonic mach number (scalar)
    r_comp= compression ratio
    theta_Bn= obliquity
    '''

    #Initialize:
    tn = 5829 #len(B['date_time'])
    l_hit = np.zeros((tn,1))
    f_min = np.zeros((tn,1))
    r_bs = np.zeros((tn,3))
    B_moments = np.zeros((tn,1))

    print("Data set size:",tn)

    #Add a unix time column to the B dataframe:
    dates = B['date_time']
    t = pd.DatetimeIndex(dates, dayfirst=True)
    t = t.astype(np.int64)/(10**9)
    B['unix_time'] = t

    #R_epoch -> float
    #R['epoch'] = R['epoch'].astype(np.int64)
    R['unix_time'] = R['epoch'].astype(np.int64)

    #Check the orientation of coordinate system:
    if(np.average(B['Bx']) > 0):
        B['By'] = -1*B['By']
        B['Bz'] = -1*B['Bz']
    

    #Create the array "l" with 1800 elements. l is used in defining r.
    ln = 1800
    l = np.zeros(ln,dtype=int)
    for index in range(len(l)):
        l[index] = index+1
    l = np.matrix(l) #1x1800 matrix

    #Define unix_time vectors to be used in the upcoming loop:
    unxtim_B = B['unix_time']
    unxtim_B = np.array(unxtim_B)
    unxtim_R = R['unix_time']
    unxtim_R = np.array(unxtim_R)

    for i in range(0,tn):

        #B_i is the i:th IMF vector from observed dataset
        B_i_row = B.iloc[i]
        B_i = np.array([B_i_row['Bx'],B_i_row['By'],B_i_row['Bz']])

        #The first index of R where time is greater than or equal to the 
        #moment of time at hand.
        #If no index found, set i_r = np.NaN
        try:
            i_r = h_findtime(unxtim_R,unxtim_B[i])
        except KeyError:
            print("KeyError at {}! Expecting a float, got something else. Skipping to the next datapoint.".format(i))
            continue
        

        #If i_r is the same as the length of unxtim_R, then we have ran out of R, which
        #means we should just exit the loop.
        if(i_r == len(unxtim_R)):
            print("Ran out of R at {}, exiting loop.".format(i))
            break


        #r_sc is simply the the coordinate vector of the sc.
        r_row = R.iloc[i_r]
        r_sc = np.array([r_row['Xgse'],r_row['Ygse'],r_row['Zgse']])

        #Convert B_1 to matrix form, so python can do linear algebra with it.
        B_i = np.matrix(B_i) #1x3 matrix

        #r is a 1800x3 matrix, which contains 1800 coordinates that lie in the
        #straight line connecting the sc and bow shock.
        r = numpy.matlib.repmat(r_sc,ln,1) + (60*numpy.matlib.repmat(B_i,ln,1)/5) + 0.05*(np.dot(l.T,B_i))

        #Calculate the equation of plane 
        f = merka05_value(nc,Vc,Bc,r)

        #Return the value that is closest to to zero and the respective index
        l_closest = np.argmin(abs(f))
        f_min[i]  = abs(f[l_closest])

        #Set this position vector to be the position of of the bow shock and
        #store the index in l_hit
        r_bs[i] = r[l_closest]
        l_hit[i] = l_closest

        #Save the moments of time in B_moments:
        B_moments[i,0] = unxtim_B[i]

        #Print out the percentage of processed data to the terminal:
        percent = 100 * i/tn
        if(i % 2 == 0):
            print("Data processed: {} %".format(np.round(percent,decimals=2)), end='\r', flush=True)

    #Refresh terminal after exiting the loop
    print("", end='\n',flush=True)


    #Once we exit the loop, cut the excess data from the endside away
    f_min = f_min[0:i+1]
    l_hit = l_hit[0:i+1]
    r_bs = r_bs[0:i+1,0:3]
    B_moments = B_moments[0:i+1]
    
    #... and update tn
    tn = len(f_min)

    
    #If f_min > 5, then no bow shock connection
    #(I don't know if 5 is an arbitrary choice) 
    for j in range(len(f_min)):
        if(f_min[j] > 5):
            f_min[j] = np.NaN
            r_bs[j] = np.NaN
            l_hit[j] = np.NaN
    
    
    #Shock normal at r_bs:
    n = merka05_normal(nc,Vc,Bc,r_bs)

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
            Bn[k] = n[k,0]*Bx[k] + n[k,1]*By[k] + n[k,2]*Bz[k]
            theta_Bn[k] = np.degrees(np.arccos(Bn[k]/Babs[k]))
            if( theta_Bn[k] > 90 ):
                theta_Bn[k] = 180 - theta_Bn[k]
        except KeyError:
            print("KeyError at {} when calculating theta_Bn! Expecting a float, got something else. Substituting NaN.".format(k))
            theta_Bn[k] = np.NaN
    

    #Construct the tn x 4 solar wind matrix
    V1 = np.ones((tn,1)) #"time"
    V2 = -400*10**3*np.ones((tn,1)) #x-component
    V3 = np.zeros((tn,1)) #y-component
    V4 = np.copy(V3)

    V = np.array([[V1],[V2],[V3],[V4]])
    V = np.matrix(V)
    V = V.T
    Vn = np.zeros(tn)
    #Dot product with velocity:
    for m in range(tn):
        Vn[m] = n[m,0]*V[m,1] + n[m,1]*V[m,2] + n[m,2]*V[m,3]

    #Local Alfven Mach number:
    mp = 1.67*10**(-27) #kg
    rho = (mp*4.5*10**6)*np.ones([tn,1]) #Is vector really necessary here?
    mu_0 = 4*np.pi*10**(-7) #Vs/Am
    gamma = 5/3
    '''
    =========================USER NOTICE========================================
    Calculating Man may give MemoryError with datasets of >20 000 datapoints.
    MemoryError arises due to lack of RAM. For my own purposes I'll be commenting
    it out.
    '''
    #Man = (Vn*np.sqrt(mu_0*rho))/(Bn*10**(-9))

    #Wind obs:
    beta = 1.4
    vs = 54*10**3
    va = 50*10**3
    vms = np.sqrt(vs**2 + va**2)
    Mms = -Vn/vms #approx.

    '''
    Compression ratio for an oblique shock, set to 2 if x > -80 Re, otherwise 1.
    This is because in the current state of this program, we treat the compression ratio
    the bow shock as a constant (which it isn't). We know that the compression ratio of
    the BS decreases as we go further along the tail, until it is 1 (not a shock anymore).

    In conclusion 80 Re is semi-arbitrary choice, just so we know which parts of space we
    are operating in.
    '''
    r_comp = np.zeros(tn)
    for p in range(tn):
        if(r_bs[p,0] > -80):
            r_comp[p] = 2.0
        else:
            r_comp[p] = 1.0
    

    #Reshape necessary lists so that np.hstack works on them
    Mms = Mms.reshape((tn,1))
    r_comp = r_comp.reshape((tn,1))
    theta_Bn = theta_Bn.reshape((tn,1))

    #If r_comp == 1, then set other values to NaN
    for p in range(tn):
        if(r_comp[p] == 1):
            f_min[p] = np.NaN
            r_bs[p] = np.NaN
            l_hit[p] = np.NaN
            Mms[p] = np.NaN
            r_comp[p] = np.NaN
            theta_Bn[p] = np.NaN
        
    
    #Add time into all the lists to be returned
    f_min = np.hstack((B_moments,f_min))
    r_bs = np.hstack((B_moments,r_bs))
    l_hit = np.hstack((B_moments,l_hit))
    Mms = np.hstack((B_moments,Mms))
    r_comp = np.hstack((B_moments,r_comp))
    theta_Bn = np.hstack((B_moments,theta_Bn))

    print("")
    print("Data analyzed succesfully!")

    return (f_min, r_bs, l_hit, Mms, r_comp, theta_Bn)

#----------------------------------------------------------------------------------

def convert_cdfs_to_dataframe(filelist, varlist, nameoftimecolumn, nameofvectorcolumn):
    #import spacepy and delorean for cdfs and datetimes
    #Remember (os.environ["CDF_LIB"] = "~/") before importing pycdf 
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
    #print("Done reading data")
        
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
    cols = dataframe.columns.tolist()
    for index, values in enumerate(column_names):
            dataframe = dataframe.rename(columns={cols[index]:column_names[index]})
    return dataframe

#----------------------------------------------------------------------------------

def readcdfdata(filename):
    '''
    This function -
    1) Reads a cdf datafile
    2) Saves the data into a pandas dataframe
    3) Replaces the dataframe's columns with new more descriptive names
    4) Needs the functions "convert_cdfs_to_dataframe()" and "new_column_names()"
       to operate
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

def writedata(f,r,l,m,theta):
    '''
    If one needs to save the analyzed data, use this function to generate a .txt
    file for that.
    Notice! This function does not write down r_comp, since it's being treated 
    as a constant in the current version of this program.
    '''

    filename = "connection_data.txt"
    dataFile = open(filename, 'w')

    dataFile.write("#time\t\tf_min\t\t\tr_bs\t\tl_hit\tMms\t\t\ttheta\n")
    for i in range(len(f)):
        dataFile.write(str(f[i,0]))
        dataFile.write("\t")
        dataFile.write(str(f[i,1]))
        dataFile.write("\t")
        dataFile.write(str(r[i,1]))
        dataFile.write("\t")
        dataFile.write(str(l[i,1]))
        dataFile.write("\t")
        dataFile.write(str(m[i,1]))
        dataFile.write("\t")
        dataFile.write(str(theta[i,1]))
        dataFile.write("\n")
    
    print("Data saved into {}.".format(filename))

    dataFile.close()

#----------------------------------------------------------------------------------

if __name__ == "__main__":
    
    #R must be read in .cdf format. readcdfdata() handles it into pd.dataframe
    #B and R are probably easiest read as pandas dataframes.
    
    R_file = "ACE_position20100405.cdf"
    B_file = "ACE_magdata_16s.txt"
    P_file = "ACE_plasma_data.txt"

    R = readcdfdata(R_file)
    #R has 6 columns: epoch, Datetime, Epoch(=Datetime), Xgse, Ygse, Zgse
    

    #====HERE IS THE NORMAL PROCEDURE FOR READING FILES===================================================

    B = pd.read_csv(B_file, skiprows=62, skipfooter=4, parse_dates=[['date','time']], names=['date','time','Bmag','Bx','By','Bz'],delimiter=r"\s+",dayfirst=True,engine='python')
    B = B.loc[B['Bmag'] >= 0]
    
    #Calculate the rolling mean values for B with pandas:
    B['Bmag'] = B.Bmag.rolling(window=40, min_periods=1).mean()
    B['Bx']   = B.Bx.rolling(window=40, min_periods=1).mean()
    B['By']   = B.By.rolling(window=40, min_periods=1).mean()
    B['Bz']   = B.Bz.rolling(window=40, min_periods=1).mean()
    

    plasmadf = pd.read_csv(P_file, skiprows=65, skipfooter=4, parse_dates=[['date','time']], names=['date','time','n','Vx','Vy','Vz'],delimiter=r"\s+",dayfirst=True,engine='python')

    #======================================================================================================


    #====CHOOSE THIS IF YOU ARE READING GEOTAIL'S DATA=====================================================
    '''
    #Modification due to geotail's magnetometric data's format:
    B['Bmag'] = 0.1*B['Bmag']
    B['Bx'] = 0.1*B['Bx']
    B['By'] = 0.1*B['By']
    B['Bz'] = 0.1*B['Bz']

    plasmadf = pd.read_csv(P_file, skiprows=0, skipfooter=0, parse_dates=[['year','month','day','hour','min','sec']], names=['year','month','day','hour','min','sec','n','Vx','Vy','Vz','Vmag','1','2','3','4','5','6','7','8'], delimiter=r"\s+")

    plasmadf.rename(index=str,columns={"year_month_day_hour_min_sec": "date_time"}, inplace=True)
    plasmadf["date_time"] = pd.to_datetime(plasmadf["date_time"],format="%Y %m %d %H %M %S.%f")
    '''
    #======================================================================================================


    #====CHOOSE THIS IF YOU ARE READING THEMIS' DATA=======================================================
    '''
    B = pd.read_csv(B_file, skiprows=1, skipfooter=4, parse_dates=['time'], names=['time','Bmag','Bx','By','Bz'],delimiter=r"\s+",dayfirst=True,engine='python')
    B = B.loc[B['Bmag'] >= 0]

    plasmadf = pd.read_csv(P_file, skiprows=71, skipfooter=4, parse_dates=[['date','time']], names=['date','time','n','del','Vx','Vy','Vz'],delimiter=r"\s+",dayfirst=True,engine='python')    
    '''
    #======================================================================================================
    
    #Define nc as average density over observing period:
    #nc = np.ones(len(B['Bmag']))
    nc = np.average(plasmadf['n'])

    #Define Vc as the average speed of solar wind over the observing period:
    #Vc = np.ones(len(B['Bmag']))
    Vc = np.average(np.sqrt(plasmadf['Vx']**2 + plasmadf['Vy']**2 + plasmadf['Vz']**2))

    #Define Bc as the average strength of IMF over the observing period:
    #Bc = np.ones(len(B['Bmag']))
    Bc = np.average(B['Bmag'])

    
    #Run the connection analysis on provided data:
    f_min, r_bs, l_hit, Mms, r_comp, theta_Bn = connection(R,B,nc,Vc,Bc)


    #After completing the analysis, write the data in a txt file:
    writedata(f_min,r_bs,l_hit,Mms,theta_Bn)
    
    
    '''
    #---------------------------------------------------------
    #SANDBOX FOR TESTING:
    rdata = {'epoch': [1.27554465*10**9],'date_time': ['05-04-2010 05:00:00.000'], 'Xgse': [250.0], 'Ygse': [0.0], 'Zgse': [0.0]}
    bdata = {'date_time': ['05-04-2010 05:00:00.000'], 'Bmag': [6.0], 'Bx': [-6.0], 'By': [0.0], 'Bz': [0.0]}

    R = pd.DataFrame(data=rdata)
    B = pd.DataFrame(data=bdata)
    nc = 4.5
    Vc = 650
    Bc = 6.0

    #---------------------------------------------------------

    print("=============f_min==================")
    print(f_min)
    print("=============r_bs==================")
    print(r_bs)
    print("=============l_hit==================")
    print(l_hit)
    print("=============Mms==================")
    print(Mms)
    print("=============r_comp==================")
    print(r_comp)
    print("=============theta_Bn==================")
    print(theta_Bn)
    
    #---------------------------------------------------------
    '''
