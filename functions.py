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
def cgaussianprofile(Peakint,Gradius,Cradius, devic=vr.def_device):
    a=(int(2*Cradius/0.0001)+1)
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
