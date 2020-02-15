import numpy as np
import matplotlib.pyplot as plt

def ceshi():
    a=np.arange(1,7,1)
    print(a)
    xuhao=np.arange(1,167,1)
    print(np.shape(xuhao))
    xuhao_l=list(xuhao)
    print(xuhao)
    np.random.shuffle(xuhao_l)
    print(xuhao_l)

def draw_test():
    data=np.loadtxt("tj.txt",dtype=float,encoding='utf-8')
    #a=np.random.randint(1,15,(85,2))
    #print(a)
    #np.savetxt("a.txt",a,fmt='%d',delimiter="\t")
    print(data)
    data=np.array(data)
    print(data.shape)
    print(data[:,0])
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("different element for peiwe_num",fontsize=20)
    plt.xlabel("peiwei-num/n",fontsize=14)
    plt.ylabel("pinlv occupation/%",fontsize=14)
    plt.plot(data[:,0],data[:,1],color="red",linewidth=3,linestyle='--',label="Cu")
    plt.plot(data[:,0],data[:,2],color="blue",linewidth=3,linestyle=':',label="Ag")
    plt.plot(data[:,0],data[:,3],color="c",linewidth=3,marker='o',label="H")
    plt.plot(data[:,0],data[:,4],color="y",linewidth=3,marker='d',label="C")
    plt.legend(loc='upper left')
    plt.xlim(0,21)
    plt.ylim(0,85)
    

    
    plt.subplot(1,2,2)
    plt.title("different element for peiwe_num",fontsize=20)
    plt.xlabel("peiwei-num/n",fontsize=14)
    plt.ylabel("pinlv occupation/%",fontsize=14)
    plt.bar(data[:,0],data[:,1],facecolor="red",label="Cu")
    plt.bar(data[:,0],data[:,2],facecolor="blue",label="Ag")
    plt.bar(data[:,0],data[:,3],facecolor="c",label="H")
    plt.bar(data[:,0],data[:,4],facecolor="y",label="C")
    plt.legend(loc='upper left')
    plt.xlim(0,21)
    plt.ylim(0,85)
    plt.show()
 
 

    
def main():
    draw_test()
    




main()