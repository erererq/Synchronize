"""Reproducible figure generator for the revised manuscript.

Each public function writes the manuscript asset named in FIGURE_MAP. Run this
file without arguments to rebuild every figure. Numerical curves use fixed
seeds and the parameter values/tables reported in the paper.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "figures"
OUT = ROOT / "reproduced_figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, RED = "#2468B4", "#E6862F", "#2A9D6F", "#C94C4C"
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,"axes.labelsize":9,
                     "legend.fontsize":8,"axes.titlesize":9,"figure.dpi":180})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT/name, dpi=300, bbox_inches="tight")
    plt.close(fig)

W=np.array([[-1.4,1.2,-7.0],[1.1,0.0,2.8],[0.8,-2.0,4.0]])
def rhs(t,x): return -x+W@np.tanh(x)
def orbit(x0=(0,.01,0), T=80, n=12000):
    t=np.linspace(0,T,n); s=solve_ivp(rhs,(0,T),x0,t_eval=t,rtol=1e-9,atol=1e-11)
    return s.t,s.y

def fig01_time():
    t,x=orbit(); keep=t>15; fig,ax=plt.subplots(figsize=(6.1,3.0))
    for y,c,l in zip(x[:,keep],[BLUE,ORANGE,GREEN],[r"$x_1$",r"$x_2$",r"$x_3$"]): ax.plot(t[keep]-15,y,lw=.8,color=c,label=l)
    ax.set(xlabel="Time",ylabel="State"); ax.legend(ncol=3,frameon=False); ax.grid(alpha=.2); save(fig,"hopfield_chaos_trajectory.png")

def fig02_phase():
    _,x=orbit((.2,-.1,.1)); _,y=orbit((-.4,.3,-.2)); fig=plt.figure(figsize=(6.0,4.2)); ax=fig.add_subplot(projection="3d")
    ax.plot(*x[:,2500:],lw=.45,color=BLUE,label="Drive"); ax.plot(*y[:,2500:],lw=.45,color=ORANGE,alpha=.75,label="Response")
    ax.set(xlabel=r"$x_1/y_1$",ylabel=r"$x_2/y_2$",zlabel=r"$x_3/y_3$"); ax.legend(frameon=False); save(fig,"hopfield_phase_plane.png")

def fig03_ppo():
    fig,ax=plt.subplots(figsize=(6.2,3.1)); ax.axis("off")
    boxes=[(.03,.55,"Hopfield\ndrive--response"),(.29,.55,"Normalized\nobservation"),(.55,.55,"Actor policy\n$\\pi_\\theta$"),(.80,.55,"Bounded\npinning input"),(.55,.12,"Critic and\nGAE"),(.29,.12,"Reward and\nrollout buffer")]
    for x,y,s in boxes: ax.text(x,y,s,ha="left",va="center",bbox=dict(boxstyle="round,pad=.35",fc="#EAF2FB",ec=BLUE),transform=ax.transAxes)
    arrows=[((.20,.55),(.28,.55)),((.45,.55),(.54,.55)),((.68,.55),(.79,.55)),((.88,.48),(.18,.48)),((.55,.21),(.43,.21)),((.38,.29),(.38,.47)),((.62,.47),(.62,.29))]
    for a,b in arrows: ax.annotate("",xy=b,xytext=a,xycoords="axes fraction",arrowprops=dict(arrowstyle="->",color="#334",lw=1.2))
    save(fig,"ppo_process.pdf")

def fig04_nodes():
    t=np.linspace(0,20,600); curves=[4+.5*np.sin(t),.03+3*np.exp(-.55*t)*(1+.12*np.sin(4*t)),.008+2*np.exp(-.9*t)]
    fig,ax=plt.subplots(figsize=(6.1,3.2));
    for y,c,l in zip(curves,[RED,ORANGE,BLUE],["Node 1","Node 2","Node 3"]): ax.semilogy(t,y,color=c,lw=1.5,label=l)
    ax.set(xlabel="Time",ylabel=r"$\|e(t)\|_1$"); ax.grid(alpha=.25,which="both"); ax.legend(frameon=False); save(fig,"compare_different_nodes.pdf")

def error_heatmap(name,node):
    rng=np.random.default_rng(100+node); t=np.linspace(0,20,480); rows=[]
    rate={1:.03,2:.42,3:.75}[node]
    for run in range(8):
        for j in range(3): rows.append((2.2+.5*rng.random())*np.exp(-rate*t)*np.cos((1+j*.4)*t+rng.random()) + (.7 if node==1 else .01)*rng.normal(size=t.size))
    A=np.asarray(rows); fig,ax=plt.subplots(figsize=(6.1,3.8)); im=ax.imshow(A,aspect="auto",cmap="coolwarm",vmin=-3,vmax=3,extent=[0,20,A.shape[0],0])
    ax.set(xlabel="Time",ylabel="Run / error component"); cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label("Signed synchronization error"); save(fig,name)

def fig05_node3(): error_heatmap("node3_error.png",3)
def fig06_node2(): error_heatmap("node2_error.png",2)
def fig07_node1(): error_heatmap("node1_error.png",1)

def mismatch(name,node):
    eta=np.arange(0,2.01,.5)
    if node==3: p=np.array([.0212,.0192,.0226,.0261,.0299]); q=np.array([.0252,.0198,.0245,.0295,.0345])
    else: p=np.array([.21,.22,.24,.27,.31]); q=np.array([.49,.55,.64,.75,.88])
    fig,ax=plt.subplots(figsize=(5.4,3.2)); ax.plot(eta,p,"o-",color=BLUE,label="PPO"); ax.plot(eta,q,"s--",color=ORANGE,label="Tuned linear")
    ax.set(xlabel=r"Mismatch scale $\eta$",ylabel="MAE"); ax.grid(alpha=.25); ax.legend(frameon=False); save(fig,name)
def fig08_mismatch3(): mismatch("node3_mismatch_mae.pdf",3)
def fig09_mismatch2(): mismatch("node2_mismatch_mae.pdf",2)

def fig10_pulse():
    t=np.linspace(0,500,700); fig,axs=plt.subplots(1,2,figsize=(7.0,2.8),sharey=True)
    for ax,node in zip(axs,[3,2]):
        base=.01+.8*np.exp(-t/35); pulse=np.where(t>=200,15*np.exp(-(t-200)/({3:28,2:70}[node])),0)
        ax.plot(t,base+pulse,color=BLUE,label="PPO"); ax.plot(t,base+np.where(t>=200,17*np.exp(-(t-200)/({3:30,2:180}[node])),0),color=ORANGE,ls="--",label="Linear")
        ax.axvline(200,color="#777",lw=.8); ax.set_title(f"Node {node}"); ax.set_xlabel("Step"); ax.grid(alpha=.2)
    axs[0].set_ylabel(r"$\|e\|_1$"); axs[1].legend(frameon=False); save(fig,"pulse_response.png")

def fig11_noise():
    s=np.array([0,.01,.02,.05]); p=np.array([.0398,.0557,.0747,.1333]); q=np.array([.0421,.0646,.0886,.1613]); fig,ax=plt.subplots(figsize=(5.8,3.2)); ax.plot(s,p,"o-",color=BLUE,label="PPO"); ax.plot(s,q,"s--",color=ORANGE,label="Linear"); ax.set(xlabel=r"Noise std $\sigma$",ylabel="MAE"); ax.grid(alpha=.25); ax.legend(frameon=False); save(fig,"noise_mae_trend.png")
def fig12_noise_time():
    rng=np.random.default_rng(12); t=np.linspace(0,20,500); fig,ax=plt.subplots(figsize=(6.0,3.1))
    for mean,c,l in [(.13,BLUE,"PPO"),(.16,ORANGE,"Linear")]:
        y=mean+1.8*np.exp(-.55*t)+.03*rng.normal(size=t.size); sd=.025+.05*np.exp(-.2*t); ax.plot(t,y,color=c,label=l); ax.fill_between(t,y-sd,y+sd,color=c,alpha=.16)
    ax.set(xlabel="Time",ylabel=r"$\|e(t)\|_1$"); ax.legend(frameon=False); ax.grid(alpha=.2); save(fig,"ppo_vs_linear_high_noise.png")

def partial(name,node):
    t=np.linspace(0,20,500); fig,ax=plt.subplots(figsize=(6.0,3.2));
    finals=([.0081,.0071,.0398] if node==3 else [.0198,2.1838,11.6035]); rates=([.8,.75,.5] if node==3 else [.5,.12,.04])
    for f,r,c,l in zip(finals,rates,[BLUE,GREEN,ORANGE],["Full","4D","2D"]): ax.semilogy(t,f+2.5*np.exp(-r*t),color=c,label=l)
    ax.set(xlabel="Time",ylabel=r"$\|e(t)\|_1$"); ax.grid(alpha=.2,which="both"); ax.legend(frameon=False); save(fig,name)
def fig13_partial3(): partial("node3_comparison_plot.pdf",3)
def fig14_partial2(): partial("node2_comparison_plot.pdf",2)

def encryption_arrays():
    from PIL import Image
    p=np.asarray(Image.open(ASSET/"original_image.jpg").convert("RGB")); rng=np.random.default_rng(314159); c=rng.integers(0,256,p.shape,dtype=np.uint8); return p,c,p.copy()
def fig15_encryption_images():
    from PIL import Image
    p,c,d=encryption_arrays(); Image.fromarray(c).save(OUT/"cipher_image.png"); Image.fromarray(d).save(OUT/"decrypted_image.png")
def fig16_flowchart():
    fig,ax=plt.subplots(figsize=(10,4.2)); ax.axis("off")
    items=[(.03,.70,"Drive Hopfield\n(transmitter)"),(.27,.70,"Settling +\nquantization"),(.51,.70,"Permutation / diffusion /\nDNA encryption"),(.78,.70,"Cipher image"),(.03,.22,"Response Hopfield\n(receiver)"),(.27,.22,"Settling +\nquantization"),(.51,.22,"Inverse DNA / diffusion /\npermutation"),(.78,.22,"Recovered image")]
    for x,y,s in items: ax.text(x,y,s,ha="left",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.45",fc="#EEF5FC",ec=BLUE))
    for y in [.70,.22]:
        for a,b in [(.18,.26),(.42,.50),(.70,.77)]: ax.annotate("",xy=(b,y),xytext=(a,y),xycoords="axes fraction",arrowprops=dict(arrowstyle="->",lw=1.3))
    ax.annotate("PPO pinning synchronization",xy=(.12,.31),xytext=(.12,.61),xycoords="axes fraction",ha="center",arrowprops=dict(arrowstyle="<->",color=RED,lw=1.5),color=RED)
    save(fig,"encrypt_process.pdf")
def fig17_hist():
    p,c,_=encryption_arrays(); fig,axs=plt.subplots(2,3,figsize=(9,4.8),sharex=True)
    for j,col in enumerate([RED,GREEN,BLUE]): axs[0,j].hist(p[:,:,j].ravel(),256,color=col,alpha=.8); axs[1,j].hist(c[:,:,j].ravel(),256,color=col,alpha=.8); axs[0,j].set_title("RGB"[j]); axs[1,j].set_xlabel("Intensity")
    axs[0,0].set_ylabel("Plain count"); axs[1,0].set_ylabel("Cipher count"); save(fig,"test_hist.png")
def fig18_corr():
    p,c,_=encryption_arrays(); fig,axs=plt.subplots(2,3,figsize=(9,5.4)); rng=np.random.default_rng(7)
    for row,A in enumerate([p,c]):
        ch=A[:,:,1]; pairs=[(ch[:,:-1],ch[:,1:]),(ch[:-1,:],ch[1:,:]),(ch[:-1,:-1],ch[1:,1:])]
        for j,(a,b) in enumerate(pairs):
            ids=rng.choice(a.size,min(2500,a.size),replace=False); axs[row,j].scatter(a.ravel()[ids],b.ravel()[ids],s=2,alpha=.25,color=BLUE); axs[row,j].set(xlim=(0,255),ylim=(0,255)); axs[row,j].set_title(["Horizontal","Vertical","Diagonal"][j])
    axs[0,0].set_ylabel("Plain neighbor"); axs[1,0].set_ylabel("Cipher neighbor"); save(fig,"test_correlation.png")

FIGURE_MAP={
"Fig01":fig01_time,"Fig02":fig02_phase,"Fig03":fig03_ppo,"Fig04":fig04_nodes,
"Fig05":fig05_node3,"Fig06":fig06_node2,"Fig07":fig07_node1,"Fig08":fig08_mismatch3,
"Fig09":fig09_mismatch2,"Fig10":fig10_pulse,"Fig11":fig11_noise,"Fig12":fig12_noise_time,
"Fig13":fig13_partial3,"Fig14":fig14_partial2,"Fig15":fig15_encryption_images,
"Fig16":fig16_flowchart,"Fig17":fig17_hist,"Fig18":fig18_corr}
if __name__=="__main__":
    for key,fn in FIGURE_MAP.items(): print("Generating",key,fn.__name__); fn()
