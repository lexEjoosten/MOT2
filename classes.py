from pyrsistent import s
import variables as vr
import functions as fn
import torch
import numpy as np
from math import pi,sqrt
from random import random
from scipy.constants import k as conk
from scipy.stats import chi





class Transition:  
    def __init__(self,DR,LL,UL,particle):
        #creating a transition requires the natural decay rate, the lower level, and the upper level, and finally the particle for which the transition applies
        #DR is the decay rate, given in Hz
        #LL is the lower level, a string in the "levels" dictionary of the particle
        #Ul is the upper level"                                                     "
        #particle must be a class with the properties assigned
        #the transition automatically calculates the M_F of the upper and lower levels based on the F in the upper and lower level designation
        self.DR=DR
        self.LL=LL
        self.UL=UL
        if (LL in particle.relevant_levels)!=True:
            particle.relevant_levels.append(LL)
        if (UL in particle.relevant_levels)!=True:
            particle.relevant_levels.append(UL)
        self.Ml=int(LL[LL.index("F")+1])
        self.Mu=int(UL[UL.index("F")+1])
        self.Energy=particle.levels[UL]-particle.levels[LL]


class species:
    def __init__(self, mass, energy_levels, isotope, nuclear_spin):
        self.mass=mass
        self.levels=energy_levels
        self.relevant_levels=[]
        self.isotope=isotope
        self.nspin=nuclear_spin
        self.dtype='species_indicator'
    
    def createstructure(self):
        self.lowestlevel=None
        self.lvlsize=0
        self.slicestart={}
        self.sliceend={}
        for i in self.relevant_levels:

            if self.lowestlevel==None:
                self.lowestlevel=i
            elif self.levels[i]<self.levels[self.lowestlevel]:
                self.lowestlevel=i
            self.slicestart[i]=self.lvlsize
            self.lvlsize+=2*int(i[i.index("F")+1])+1
            self.sliceend[i]=self.lvlsize


        

class tens:
    #this class defines some default tensors which will be used in testing
    x=torch.tensor([1,0,0],device=vr.def_device,dtype=torch.get_default_dtype())
    y=torch.tensor([0,1,0],device=vr.def_device,dtype=torch.get_default_dtype())
    z=torch.tensor([0,0,1],device=vr.def_device,dtype=torch.get_default_dtype())
    t0=torch.tensor([0,0,0],device=vr.def_device,dtype=torch.get_default_dtype())


class laserbeam:
    def __init__(self,passes_through, direction, wavelength, profile,profilex=None,profiley=None,Cutoff=None,def_device=vr.def_device, profile_size=(0.0001,0.0001)):
        #passes_through gives one point where the beam passes through (default=[0,0,0])
        #direction gives the direction in which the beam moves
        #wavelength gives the wavelength of the light in the beam.
        #profile is given as a matrix of intensities with the center of the matrix being the center of the beam (expected type is np matrix)
        #profile is measured with grid squares of default size .1*.1 mm
        #profile indicates the beam shape and intensities, not wavelenght profile.

        self.k=2*pi/wavelength
        direction=torch.tensor(direction,device=def_device,dtype=torch.get_default_dtype())
        self.dir=torch.divide(direction,torch.norm(direction,dim=0))
        if passes_through==0:
            self.P0=torch.tensor([0,0,0],device=def_device,dtype=torch.get_default_dtype())
        else:
            self.P0=passes_through-torch.dot(passes_through,self.dir)
        self.kvect=self.dir*self.k
        self.profile=profile
        self.cutoff=Cutoff
        self.psize=profile_size


        if profilex==None:
            self.profilex= torch.divide(torch.cross(tens.x+tens.y+tens.z,self.dir),torch.norm(torch.cross(tens.x+tens.y+tens.z,self.dir))) 
            #The particular choice for the basis vector against which the profile axes are defined is because if the direction is in either x,y, or z,
            # and the profile axes are defined as a cross product with any of these directions, this would obviously cause issues.
            #This particular choice will not cause issues for directions not with this particular arrangement, because
            # the profile should tend to be symmetric, and hence the particular alignment of the profile axes should not matter. 
            #If it does matter, profile axes can be defined manually. Which would also be fine
        else:
            self.profilex=profilex
        if profiley==None:
            self.profiley= torch.cross(self.profilex,self.dir)
            #This arrangement ensures that the profile axes are aligned orthogonally to eachother and the direction of the laserbeam.
            #Normalization is already built in as a result of the definitions of profilex, and dir.
        else:
            self.profiley=profiley
        
    
    def intensity(self,positions):
        #this should return a tensor of intensities from the laserbeam profile.
    	
        I=torch.zeros(positions.shape[0],device=positions.device,dtype=torch.get_default_dtype())


        
        if len(positions.shape)==0:
            #makes sure the position has the right shape, only relevant if a single position is inputted
            positions.unsqueeze(0)
       
        R=fn.disttobeam(positions,self)
        #R encodes the distance to the beam
        
        if self.cutoff!=None:
            R=(R<=self.cutoff)
        else:
            R=(R<=(sqrt(2)*self.profile.shape[0])*max(self.psize))
        #R becomes a truth table stating whether each particle should be considered, depending on whether the particle is outside of the cutoff range
        #Or, if no cutoff is present, whether the particle could be within range of the intensity profile defined by self.profile

        R=((R == True).nonzero()).squeeze()  
        #R becomes a tensor containing the indices of the particles affectable by the laser

        Px=torch.matmul(positions[R,:]-self.P0,self.profilex.unsqueeze(1)).squeeze()
        Py=torch.matmul(positions[R,:]-self.P0,self.profiley.unsqueeze(1)).squeeze()
        #Px,Py are the positions of the particles along the profilex and profiley directions
        Px=((Px+(self.psize[0]*self.profile.shape[0]/2))/self.psize[0]).int()
        Py=((Py+(self.psize[1]*self.profile.shape[1]/2))/self.psize[1]).int()
        (xmax,ymax)=self.profile.shape
        #Final check ensuring the particle has a defined position within the profile. The Cutoff check should automatically ensure this, 
        #and this check is needed for non-cutoff lasers
        P=((0<=Px) & (Px<xmax) & (0<=Py) & (Py<ymax)).nonzero().squeeze()
        Px=Px[P].long()
        Py=Py[P].long()
        R=R[P]
        P=self.profile[Px,Py]
        I[R]=P.to(torch.get_default_dtype())


        return I
    

#Each particle species is its own class, this effectively means that modelling multiple particle species requires cycling through the species, this is inefficient,
# but the operations performed on each particle species will be different as they will have different transitions. Thus I see no workaround yet.

class particles:
    #the particles class contains the data for the particles included in the model, this data includes particle position, velocity, species, and energy level occupations
    def __init__(self, species):
        self.species=species
    
    
    def create(self, positions,velocities):
        self.x=positions
        self.v=velocities
        #need species to be represented as a numpy array, as torch tensors cant include general pointers

        

        self.levels=torch.zeros((positions.shape[0],self.species.lvlsize),device=self.x.device)
        self.levels[:,self.species.slicestart[self.species.lowestlevel]:self.species.sliceend[self.species.lowestlevel]]=1/(2*int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])+1)*torch.ones((positions.shape[0],int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])),device=self.x.device)
        

    def createbyT(self, N,T=300,R=0.01,def_device=vr.def_device):
        #produces a cloud of N particles at radius R, as though particles wander into this sphere and come into the simulated surface


        #randomly distribute points over a shell at radius R. 
        self.x=fn.randthreevecs(N,def_device)
        # Need to use the chi4 distribution as we're not interested in all the velocies in the gas chamber(which would be maxwell distribution (chi3)) but
        #  instead only those which pass te boundary, which is effusion, which must be multiplied by v_n, where v_n is the velocity into the  sphere. 
        # This also requires the modification of the direction distribution. But doing this correctly allows for a simplified insertion of particles to the model
        #
        # This mode of addition is neccesarily based off of the ideal gas model.
        v_therm=sqrt(2*conk*T/self.species.mass)
        #see notes
        vel=chi.rvs(4,size=N,scale=v_therm)*2/pi
        phi=torch.zeros((N),device=def_device).uniform_(0,2*np.pi)
        theta=torch.arcsin(torch.sqrt(torch.zeros((N),device=def_device).uniform_()))
        
        nx=(torch.outer(self.x[:,2],torch.tensor([0,1,0],device=def_device))-torch.outer(self.x[:,1],torch.tensor([0,0,1],device=def_device))).T
        ny=(torch.outer(self.x[:,2],torch.tensor([1,0,0],device=def_device))-torch.outer(self.x[:,0],torch.tensor([0,0,1],device=def_device))).T
        nx=torch.divide(nx,torch.norm(nx,dim=0))
        ny=torch.divide(ny,torch.norm(ny,dim=0))
        self.v=(torch.tensor(vel,device=def_device)*(torch.mul(torch.cos(theta),self.x.T)+torch.mul(torch.sin(theta),(torch.mul(torch.cos(phi),nx)+torch.mul(torch.sin(phi),ny))))).T
        
        self.x=R*self.x
        

        self.levels=torch.zeros((N,self.species.lvlsize),device=self.x.device)
        self.levels[:,self.species.slicestart[self.species.lowestlevel]:self.species.sliceend[self.species.lowestlevel]]=1/(2*int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])+1)*torch.ones((N,int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])),device=self.x.device)
        
    def add(self, positions,velocities):
        self.x=torch.cat((self.x,positions))
        self.v=torch.cat((self.v,velocities))
        #need species to be represented as a numpy array, as torch tensors cant include general pointers
        
        levels=torch.zeros((positions.shape[0],self.species.lvlsize),device=self.x.device)
        levels[:,self.species.slicestart[self.species.lowestlevel]:self.species.sliceend[self.species.lowestlevel]]=1/(2*int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])+1)*torch.ones((positions.shape[0],int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])))

        self.levels=torch.cat((self.levels,levels))

    def addbyT(self, N,T=300,R=0.01,def_device=vr.def_device):
        #produces a cloud of N particles at radius R, as though particles wander into this sphere and come into the simulated surface


        #randomly distribute points over a shell at radius R. 
        x=fn.randthreevecs(N,def_device)
        # Need to use the chi4 distribution as we're not interested in all the velocies in the gas chamber(which would be maxwell distribution (chi3)) but
        #  instead only those which pass te boundary, which is effusion, which must be multiplied by v_n, where v_n is the velocity into the  sphere. 
        # This also requires the modification of the direction distribution. But doing this correctly allows for a simplified insertion of particles to the model
        #
        # This mode of addition is neccesarily based off of the ideal gas model.
        v_therm=sqrt(2*conk*T/self.species.mass)
        #see notes
        vel=chi.rvs(4,size=N,scale=v_therm)*2/pi
        phi=torch.zeros((N),device=def_device).uniform_(0,2*np.pi)
        theta=torch.arcsin(torch.sqrt(torch.zeros((N),device=def_device).uniform_()))
        nx=(torch.outer(x[:,2],torch.tensor([0,1,0],device=def_device))-torch.outer(x[:,1],torch.tensor([0,0,1],device=def_device))).T
        ny=(torch.outer(x[:,2],torch.tensor([1,0,0],device=def_device))-torch.outer(x[:,0],torch.tensor([0,0,1],device=def_device))).T
        nx=torch.divide(nx,torch.norm(nx,dim=0))
        ny=torch.divide(ny,torch.norm(ny,dim=0))
        v=(torch.tensor(vel,device=def_device)*(torch.mul(torch.cos(theta),x.T)+torch.mul(torch.sin(theta),(torch.mul(torch.cos(phi),nx)+torch.mul(torch.sin(phi),ny))))).T
        
        x=R*x
        self.x=torch.cat((self.x,x))
        self.v=torch.cat((self.v,v))

        levels=torch.zeros((N,self.species.lvlsize),device=self.x.device)
        levels[:,self.species.slicestart[self.species.lowestlevel]:self.species.sliceend[self.species.lowestlevel]]=1/(2*int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])+1)*torch.ones((N,int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])))

        self.levels=torch.cat((self.levels,levels))

        

    def timestepadd(self,dt=0.00001 ,T=300,P=3e-7,R=0.01,def_device=vr.def_device):
        #produces a cloud of N particles at radius R, as though particles wander into this sphere and come into the simulated surface
        N=dt*4*pi*R**2*P/sqrt(2*pi*self.species.mass*conk*T)
        Nmin=int(N)
        if (N-Nmin)>=random():
            N=Nmin+1
        else:
            N=Nmin

        #randomly distribute points over a shell at radius R. 
        x=fn.randthreevecs(N,def_device)
        # Need to use the chi4 distribution as we're not interested in all the velocies in the gas chamber(which would be maxwell distribution (chi3)) but
        #  instead only those which pass te boundary, which is effusion, which must be multiplied by v_n, where v_n is the velocity into the  sphere. 
        # This also requires the modification of the direction distribution. But doing this correctly allows for a simplified insertion of particles to the model
        #
        # This mode of addition is neccesarily based off of the ideal gas model.
        v_therm=sqrt(2*conk*T/self.species.mass)
        #see notes
        vel=chi.rvs(4,size=N,scale=v_therm)*2/pi
        phi=torch.zeros((N),device=def_device).uniform_(0,2*np.pi)
        theta=torch.arcsin(torch.sqrt(torch.zeros((N),device=def_device).uniform_()))
        nx=(torch.outer(x[:,2],torch.tensor([0,1,0],device=def_device))-torch.outer(x[:,1],torch.tensor([0,0,1],device=def_device))).T
        ny=(torch.outer(x[:,2],torch.tensor([1,0,0],device=def_device))-torch.outer(x[:,0],torch.tensor([0,0,1],device=def_device))).T
        nx=torch.divide(nx,torch.norm(nx,dim=0))
        ny=torch.divide(ny,torch.norm(ny,dim=0))
        v=(torch.tensor(vel,device=def_device)*(torch.mul(torch.cos(theta),x.T)+torch.mul(torch.sin(theta),(torch.mul(torch.cos(phi),nx)+torch.mul(torch.sin(phi),ny))))).T
        
        x=R*x
        self.x=torch.cat((self.x,x))
        self.v=torch.cat((self.v,v))
    
        
        levels=torch.zeros((N,self.species.lvlsize),device=self.x.device)
        levels[:,self.species.slicestart[self.species.lowestlevel]:self.species.sliceend[self.species.lowestlevel]]=1/(2*int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])+1)*torch.ones((N,int(self.species.lowestlevel[self.species.lowestlevel.index("F")+1])))

        self.levels=torch.cat((self.levels,levels))
