"""Report support for Direction 2 (Zekun Wang). Reuses build() from ticket_d2_study.py
(copied verbatim) without re-running its study loop; computes truncation checks,
symmetrisation checks, scaled residuals, and draws Palatino figures."""
import sys, glob, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, csv
import matplotlib; matplotlib.use("Agg")
from matplotlib import font_manager as fm, pyplot as plt
src=open("ticket_d2_study.py").read().split("rows=[]")[0]; exec(src)   # defines ue, build
for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf")+glob.glob("/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf"): fm.fontManager.addfont(f)
matplotlib.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Pagella"],"mathtext.fontset":"custom",
 "mathtext.rm":"TeX Gyre Pagella","mathtext.it":"TeX Gyre Pagella:italic","mathtext.bf":"TeX Gyre Pagella:bold",
 "axes.spines.top":False,"axes.spines.right":False,"font.size":9,"axes.linewidth":0.6,"lines.linewidth":1.0,
 "lines.markersize":4,"legend.frameon":False,"pdf.fonttype":42})
NAVY,BRICK,SAGE,GREY="#2b3f5c","#9c4a3c","#6b8e6b","#7a7a7a"
FIG="../report/figures/"; out=open("../results/d2_extra_log.txt","w")
def P(*a): print(*a); print(*a,file=out)
# 1. truncation check: apply A to exact nodal values
rows=[]
for beta,lab in [(2.0,"Robin"),(0.0,"Neumann")]:
  for cl in ["first","ghost"]:
    for N in [16,32,64,128,256]:
      A,F,Ue,h,I,J=build(N,beta,cl); tau=A@Ue-F; b=I==N
      U=spla.spsolve(A.tocsc(),F); r=F-A@U
      D=np.where(b,0.5,1.0) if cl=="ghost" else np.where(b,1/h,1.0)
      S=sp.diags(D)@A; etaS=spla.norm(S-S.T)/spla.norm(S)
      # scaled residual: row-wise relative to |A||U|+|F|
      sres=np.max(np.abs(r)/(abs(A)@np.abs(U)+np.abs(F)))
      rows.append(dict(bc=lab,closure=cl,N=N,h=h,tauB=np.abs(tau[b]).max(),tauI=np.abs(tau[~b]).max(),
          etaA=spla.norm(A-A.T)/spla.norm(A),etaS=etaS,scaled_res=sres))
      P(f"{lab:7s} {cl:5s} N={N:3d} max|tau_B|={rows[-1]['tauB']:.3e} (/h={rows[-1]['tauB']/h:.4f}) max|tau_I|={rows[-1]['tauI']:.3e} (/h^2={rows[-1]['tauI']/h**2:.4f}) eta(A)={rows[-1]['etaA']:.4f} eta(DA)={etaS:.1e} scaled_res={sres:.1e}")
with open("../results/d2_extra.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
P("predicted |tau_B|/h: first 0.5*e^2(1+y) max ->", 0.5*np.e**2, " ghost (1/3)e^2 ->", np.e**2/3, "; |tau_I|/h^2 -> e^2/6 =",np.e**2/6)
# 2. CG on symmetrised systems, both closures, N=64
for beta,lab in [(2.0,"Robin"),(0.0,"Neumann")]:
  for cl in ["first","ghost"]:
    A,F,Ue,h,I,J=build(64,beta,cl); b=I==64; D=np.where(b,0.5,1.0) if cl=="ghost" else np.where(b,1/h,1.0)
    S=(sp.diags(D)@A).tocsr(); G=D*F; Ud=spla.spsolve(A.tocsc(),F); it=[0]
    x,info=spla.cg(S,G,rtol=1e-12,maxiter=20000,callback=lambda xk: it.__setitem__(0,it[0]+1))
    lmin=spla.eigsh(S,k=1,sigma=0,which="LM",return_eigenvectors=False)[0]
    P(f"CG {lab} {cl} N=64: info={info} iters={it[0]} |x-direct|_inf={np.abs(x-Ud).max():.2e} lambda_min(DA)={lmin:.4f}")
# 3. figures
class _T:
    def __init__(s,rows): s.r=rows
    def __getattr__(s,k): return np.array([x[k] for x in s.r])
    def sel(s,**kw): return _T([x for x in s.r if all(x[a]==b for a,b in kw.items())])
d=_T([{k:(float(v) if k not in("bc","closure") else v) for k,v in x.items()} for x in csv.DictReader(open("ticket_d2_results.csv"))])
fig,ax=plt.subplots(1,2,figsize=(6.4,2.7),sharey=True)
for a,bc in zip(ax,["Neumann","Robin"]):
  for cl,c,m in [("first",BRICK,"o"),("ghost",NAVY,"s")]:
    s=d.sel(bc=bc,closure=cl)
    a.loglog(s.h,s.L2,m+"-",color=c,mfc="white",label=f"{cl}, $L^2_h$")
    a.loglog(s.h,s.Linf,m+"--",color=c,label=f"{cl}, $L^\\infty$")
  hh=np.array([s.h.min(),s.h.max()])
  a.loglog(hh,0.25*hh,":",color=GREY); a.loglog(hh,0.02*hh**2,":",color=GREY)
  a.text(hh[0]*1.1,0.25*hh[0]*0.45,"slope 1",color=GREY,fontsize=7); a.text(hh[0]*1.1,0.02*hh[0]**2*0.3,"slope 2",color=GREY,fontsize=7)
  a.set_xticks(s.h); a.set_xticklabels([f"1/{int(round(1/v))}" for v in s.h]); a.minorticks_off()
  a.set_title(f"{bc} on $x=1$",fontsize=9); a.set_xlabel("$h$")
ax[0].set_ylabel("error"); ax[1].legend(fontsize=7,loc="lower right")
fig.tight_layout(); fig.savefig(FIG+"convergence.pdf"); plt.close(fig)
# tau scaling figure
t=_T(rows); fig,a=plt.subplots(figsize=(3.4,2.6))
for cl,c,m in [("first",BRICK,"o"),("ghost",NAVY,"s")]:
  s=t.sel(bc="Robin",closure=cl); a.loglog(s.h,s.tauB,m+"-",color=c,label=f"boundary rows, {cl}")
s=t.sel(bc="Robin",closure="ghost"); a.loglog(s.h,s.tauI,"^-",color=SAGE,label="interior rows")
a.set_xticks(s.h); a.set_xticklabels([f"1/{int(round(1/v))}" for v in s.h]); a.minorticks_off(); a.set_xlabel("$h$"); a.set_ylabel(r"$\max|\tau|$"); a.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG+"truncation.pdf"); plt.close(fig)
# spy + error field
fig,ax=plt.subplots(1,3,figsize=(6.6,2.4))
A,F,Ue,h,I,J=build(8,2.0,"ghost"); ax[0].spy(A,markersize=2,color=NAVY); ax[0].set_title("ghost, $N=8$",fontsize=9)
A,F,Ue,h,I,J=build(8,2.0,"first"); ax[1].spy(A,markersize=2,color=BRICK); ax[1].set_title("first-order, $N=8$",fontsize=9)
for a in ax[:2]: a.tick_params(labelsize=7)
A,F,Ue,h,I,J=build(64,2.0,"ghost"); U=spla.spsolve(A.tocsc(),F)
E=np.abs(U-Ue).reshape(63,64); im=ax[2].imshow(E,origin="lower",extent=[h/2,1+h/2,h/2,1-h/2],cmap="cividis",aspect="equal")
ax[2].set_title("$|U-u^*|$, ghost Robin, $N=64$",fontsize=9); ax[2].set_xlabel("$x$"); ax[2].tick_params(labelsize=7)
cb=fig.colorbar(im,ax=ax[2],fraction=0.046); cb.ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(FIG+"spy_error.pdf"); plt.close(fig)
