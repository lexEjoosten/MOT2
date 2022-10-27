from torch import Tensor
import numpy as np
from numpy import cos,sin,pi,sqrt
import torch
import variables as vr



#creates a set of randomized normalized three-vectors
def randthreevecs(N=5000,def_device=vr.def_device):
    z=1-2*torch.zeros((N),device=def_device).uniform_()
    phi=torch.zeros((N),device=def_device).uniform_(0,2*np.pi)
    norm=torch.zeros((N,3),device=def_device)
    norm[:,0]=torch.sqrt(1-z**2)*torch.cos(phi)
    norm[:,1]=torch.sqrt(1-z**2)*torch.sin(phi)
    norm[:,2]=z
    return(norm)


#creates a 3x3 rotation matrix
def rot(alpha=0,beta=0,gamma=0,def_device=vr.def_device):
    a=Tensor([[cos(gamma),-sin(gamma),0],[sin(gamma),cos(gamma),0],[0,0,1]]).to(def_device)
    b=Tensor([[cos(beta),0,sin(beta)],[0,1,0],[-sin(beta),0,cos(beta)]]).to(def_device)
    c=Tensor([[1,0,0],[0,cos(alpha),-sin(alpha)],[0,sin(alpha),cos(alpha)]]).to(def_device)
    return torch.mm(torch.mm(a,b),c)

#creates a (nx3x3) rotation matrix.
def Rot(c,b,a):
    A=torch.stack([
        torch.stack([torch.cos(a),-torch.sin(a),torch.zeros(len(a)).to(vr.def_device)]),
        torch.stack([torch.sin(a),torch.cos(a),torch.zeros(len(a)).to(vr.def_device)]),
        torch.stack([torch.zeros(len(a)).to(vr.def_device),torch.zeros(len(a)).to(vr.def_device),torch.ones(len(a)).to(vr.def_device)])
    ]).transpose(0,2).transpose(1,2)
    
    B=torch.stack([
        torch.stack([torch.cos(b),torch.zeros(len(b)).to(vr.def_device),torch.sin(b)]),
        torch.stack([torch.zeros(len(b)).to(vr.def_device),torch.ones(len(b)).to(vr.def_device),torch.zeros(len(b)).to(vr.def_device)]),
        torch.stack([-torch.sin(b),torch.zeros(len(b)).to(vr.def_device), torch.cos(b)])
    ]).transpose(0,2).transpose(1,2)
    
    C=torch.stack([
        torch.stack([torch.ones(len(b)).to(vr.def_device),torch.zeros(len(b)).to(vr.def_device),torch.zeros(len(b)).to(vr.def_device)]),
        torch.stack([torch.zeros(len(b)).to(vr.def_device),torch.cos(c),-torch.sin(c)]),
        torch.stack([torch.zeros(len(b)).to(vr.def_device),torch.sin(c),torch.cos(c)])
    ]).transpose(0,2).transpose(1,2)
    
    return torch.matmul(A,torch.matmul(B,C))



#creates a set of WignerD rotation matrices (unused)
def WignerD(B):
    U = torch.stack([
                torch.stack([torch.cos(B/2)**2, -torch.sin(B)/sqrt(2), torch.sin(B/2)**2]),
                torch.stack([torch.sin(B)/sqrt(2), torch.cos(B), -torch.sin(B)/sqrt(2)]),
                torch.stack([torch.sin(B/2)**2, torch.sin(B)/sqrt(2), torch.cos(B/2)**2])]).transpose(0,2)
    return U

#translates a set of polar vectors to a set of cartesian vectors, (or the imaginary and real components thereof.)
def poltocart(ep):
    epr=torch.matmul(Tensor([[-1/sqrt(2),0,1/sqrt(2)],[0,0,0],[0,1,0]]).to(vr.def_device).unsqueeze(0).repeat(ep.shape[0],1,1),ep)
    epc=torch.matmul(Tensor([[0,0,0],[1/sqrt(2),0,1/sqrt(2)],[0,0,0]]).to(vr.def_device).unsqueeze(0).repeat(ep.shape[0],1,1),ep)
    
    
    
    return epr,epc



def carttopol(ep):
    #This translates the real and imaginary components of a cartesian vector into a polar vector.
    epr=ep[0]
    epc=ep[1]
    epr1=torch.matmul(Tensor([[-1/sqrt(2),0,0],[0,0,1],[1/sqrt(2),0,0]]).to(vr.def_device).unsqueeze(0).unsqueeze(1).repeat(epr.shape[0],epr.shape[1],1,1),epr)
    epr2=torch.matmul(Tensor([[0,1/sqrt(2),0],[0,0,0],[0,1/sqrt(2),0]]).to(vr.def_device).unsqueeze(0).unsqueeze(1).repeat(epr.shape[0],epr.shape[1],1,1),epc)
    
    epc1=torch.matmul(Tensor([[-1/sqrt(2),0,0],[0,0,1],[1/sqrt(2),0,0]]).to(vr.def_device).unsqueeze(0).unsqueeze(1).repeat(epr.shape[0],epr.shape[1],1,1),epc)
    epc2=torch.matmul(Tensor([[0,-1/sqrt(2),0],[0,0,0],[0,-1/sqrt(2),0]]).to(vr.def_device).unsqueeze(0).unsqueeze(1).repeat(epr.shape[0],epr.shape[1],1,1),epr)
    
    epr=epr1+epr2
    epc=epc1+epc2
    
    ep=torch.norm(torch.cat((epr.unsqueeze(0),epc.unsqueeze(0)),0),dim=0)
    return(ep)

    
    

def RatesbyBeam(u,l,Pa,En,Rb):
    s=En.Intensities(Pa.x)
    den=(1+4*(En.fulldtun(l,u)/Rb.Gamma)**2+s)
    Rate=s*Rb.Gamma/2*En.eploc(Pa.x)[:,:,(1+(l-u)),0]/den
    Rate=Rate*torch.sqrt(Rb.BranRat[l+2,u+3])
    return Rate


#function for creating a gaussian intensity profile
def cgaussianprofile(Peakint,Gradius,Cradius, devic=vr.def_device,size=0.0001):
    a=(int(2*Cradius/size)+1)
    profilex=np.zeros((a,1))
    profilex[:,0]=np.linspace(-Cradius,Cradius,a)
    profiley=np.zeros((1,a))
    profiley[0,:]=np.linspace(-Cradius,Cradius,a)
    profilex=Peakint*np.exp(-(profilex**2)/(2*Gradius**2))
    profiley=np.exp(-(profiley**2)/(2*Gradius**2))
    profile=torch.tensor(np.matmul(profilex,profiley),device=devic)
    return profile


#function finding the distance between a point and a 
def disttobeam(point,laserbeam):
    x_0=laserbeam.P0
    y=laserbeam.dir
    
    if(len(point.shape))==1:
        point=point.unsqueeze(0)
    return torch.norm(x_0-point-torch.mul(y,(x_0-point))*y,dim=1)

def AHh(x,A):
    return (torch.matmul(x,torch.diag(torch.tensor([A,A,-2*A],device=x.device,dtype=x.dtype))))

'''
    def Veldtun(v):
        #finds the velocity detunement
        kv = torch.inner(Environment.Lk,v)
        return -kv*Environment.Kmag

    def Bdtun(Ml,Mu,x):
        #returns the B-field detunement of a particular point.
        return 8.7941e10*torch.sqrt(torch.sum(torch.square(Environment.BaHH*x),1))*(Rubidium.gl*Ml-Rubidium.gu*Mu)
    
    def eploc(x):
        #returns the local polarization vector.
        R1=Environment.Brot(Environment.BaHH,x).unsqueeze(1).repeat(1,6,1,1)
        epr=Environment.LpolcartR.unsqueeze(0).repeat(x.shape[0],1,1,1)
        epc=Environment.LpolcartI.unsqueeze(0).repeat(x.shape[0],1,1,1)
        epr=torch.matmul(R1,epr)
        epc=torch.matmul(R1,epc)
        ep=carttopol([epr,epc])
        ep=ep
        return ep
        
    
        
    def Brot(B,x):
        #This returns the rotation vector needed to rotate the polarization of the beams onto the axis defined at the location x, by the field B.
        u=F.normalize(torch.matmul(Tensor([[0,1,0],[-1,0,0],[0,0,0]]).to(vr.base_device).unsqueeze(0).repeat(x.shape[0],1,1),(torch.matmul(Environment.aHHassym,(B*x).transpose(0,1)).transpose(0,1)).unsqueeze(2)).squeeze())
        W=torch.inner(Tensor([[[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0]]]).to(vr.base_device),u).transpose(1,2).transpose(0,1)

        phi=-torch.arccos(F.normalize(x)[:,2])
        
        Id=Tensor([[1,0,0],[0,1,0],[0,0,1]]).to(vr.base_device).unsqueeze(0).repeat(x.shape[0],1,1)
        
        R=Id+(W.transpose(0,2)*torch.sin(phi)).transpose(0,2)+((torch.matmul(W,W)).transpose(0,2)*(2*torch.sin(phi/2)**2)).transpose(0,2)
        
        
        
        return R
    
    
    
    
    
    def fulldtun(Ml,Mu,dop,zee):
        #this is a function which combines the previous functions
        return (Environment.dtun-dop*Environment.Veldtun(particles.v)-zee*Environment.Bdtun(Ml,Mu,particles.x)).transpose(0,1)
'''


def eploc(x,Environment,def_device=vr.def_device):
    #This function takes in a set of positions, and the environment. And returns the local polarizations of the laserbeams as measured along
    #the quantization axis defined by the local magnetic field.
    dirs=torch.zeros((len(Environment.laserbeams),3))
    eps=torch.zeros((len(Environment.laserbeams),3))
    for i in range(len(Environment.laserbeams)): 
        dirs[i]=Environment.laserbeams[i].dir
        eps[i]=Environment.laserbeams[i].pol
    eps=eps.unsqueeze(-1)
    Bloc=Environment.B(x)
    nBloc=torch.div(Bloc,torch.linalg.norm(Bloc,dim=1).unsqueeze(1))
    v=torch.cross(nBloc.unsqueeze(1).repeat(1,len(Environment.laserbeams),1),dirs.unsqueeze(0).repeat(nBloc.shape[0],1,1))
    

    #this calculates the sin(theta) and 1-cos(theta) factors needed for the calculation of the rotation matrix.
    s=torch.linalg.norm(v,dim=2).unsqueeze(-1).unsqueeze(-1).repeat(1,1,3,3)
    c=(torch.ones((Bloc.shape[0],dirs.shape[0]),device=def_device)-torch.sum(torch.mul(nBloc.unsqueeze(1).repeat(1,dirs.shape[0],1),dirs.unsqueeze(0).repeat(Bloc.shape[0],1,1)),dim=2)).unsqueeze(-1).unsqueeze(-1).repeat(1,1,3,3)
    
    #the following is to deal with the edge case that the direction of one of the laserbeams is paralel to that of one of the B fields.
    k=torch.linalg.norm(v,dim=2)
    k=(k==torch.zeros(k.shape,device=k.device,dtype=k.dtype)).to(k.dtype)
    k=k.unsqueeze(-1).repeat(1,1,3)*1e-10
    v=torch.cross(nBloc.unsqueeze(1).repeat(1,len(Environment.laserbeams),1)+k,dirs.unsqueeze(0).repeat(nBloc.shape[0],1,1))
    k=None

    #theta=(torch.arccos(torch.sum(torch.mul(nBloc.unsqueeze(1).repeat(1,dirs.shape[0],1),tens.z.unsqueeze(0).unsqueeze(0).repeat(Bloc.shape[0],dirs.shape[0],1)),dim=2))-torch.arccos(torch.sum(torch.mul(dirs.unsqueeze(0).repeat(Bloc.shape[0],1,1),tens.z.unsqueeze(0).unsqueeze(0).repeat(Bloc.shape[0],dirs.shape[0],1)),dim=2)))
    theta=torch.arccos(torch.sum(torch.mul(nBloc.unsqueeze(1).repeat(1,dirs.shape[0],1),dirs.unsqueeze(0).repeat(Bloc.shape[0],1,1)),dim=2))
    c=torch.cos(theta)
    s=torch.sin(theta)
    Wignery=torch.zeros((theta.shape[0],theta.shape[1],3,3),dtype=theta.dtype,device=theta.device)
    Wignery[:,:,0,0]=1/2*(torch.ones(c.shape,device=c.device,dtype=c.dtype)+c)
    Wignery[:,:,0,1]=-1/sqrt(2)*s   
    Wignery[:,:,0,2]=1/2*(torch.ones(c.shape,device=c.device,dtype=c.dtype)-c)
    Wignery[:,:,1,0]=1/sqrt(2)*s   
    Wignery[:,:,1,1]=c
    Wignery[:,:,1,2]=-1/sqrt(2)*s   
    Wignery[:,:,2,0]=1/2*(torch.ones(c.shape,device=c.device,dtype=c.dtype)-c)
    Wignery[:,:,2,1]=1/sqrt(2)*s   
    Wignery[:,:,2,2]=1/2*(torch.ones(c.shape,device=c.device,dtype=c.dtype)+c)
    
    Wignerx=torch.linalg.inv(Wignery)

    eploc=torch.matmul(Wignery,eps)
    return torch.abs(eploc)