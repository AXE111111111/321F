"""Week 6 ticket support: Direction 2 mixed BC convergence study (both closures, Robin beta=2 and Neumann).
-Lap u=f on (0,1)^2, u*=e^{x+y}; Dirichlet on x=0,y=0,y=1; on x=1: u_x+beta u=b. h=1/N, unknowns i=1..N, j=1..N-1."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, csv
ue=lambda x,y: np.exp(x+y)
def build(N,beta,closure):
    h=1/N; idx=lambda i,j:(j-1)*N+(i-1); M=N*(N-1)
    r,c,v=[],[],[]; F=np.zeros(M)
    b=lambda y:(1+beta)*np.exp(1+y)
    for j in range(1,N):
        y=j*h
        for i in range(1,N+1):
            k=idx(i,j); x=i*h
            if i==N and closure=="first":
                r+= [k,k]; c+=[k,idx(N-1,j)]; v+=[1/h+beta,-1/h]; F[k]=b(y); continue
            diag=4/h**2; F[k]=-2*np.exp(x+y)
            if i==N: diag+=2*beta/h; F[k]+=2*b(y)/h
            r.append(k); c.append(k); v.append(diag)
            nb=[(i-1,j,1.0),(i+1,j,1.0),(i,j-1,1.0),(i,j+1,1.0)]
            if i==N: nb=[(i-1,j,2.0),(i,j-1,1.0),(i,j+1,1.0)]
            for (p,q,w) in nb:
                if p==0 or q==0 or q==N: F[k]+=w*ue(p*h,q*h)/h**2
                else: r.append(k); c.append(idx(p,q)); v.append(-w/h**2)
    A=sp.csr_matrix((v,(r,c)),shape=(M,M))
    I,J=np.meshgrid(np.arange(1,N+1),np.arange(1,N),indexing="xy")
    Ue=ue(I.ravel()*h,J.ravel()*h); return A,F,Ue,h,I.ravel(),J.ravel()
rows=[]
for beta,lab in [(2.0,"Robin"),(0.0,"Neumann")]:
  for cl in ["first","ghost"]:
    prev=None
    for N in [16,32,64,128,256]:
        A,F,Ue,h,I,J=build(N,beta,cl); U=spla.spsolve(A.tocsc(),F); e=np.abs(U-Ue)
        w=np.where(I==N,0.5,1.0); L2=np.sqrt(h**2*np.sum(w*e**2)); Li=e.max(); kmax=e.argmax()
        bnd=e[I==N].max(); intr=e[I<N].max()
        eta=spla.norm(A-A.T)/spla.norm(A)
        rows.append(dict(bc=lab,closure=cl,N=N,h=h,L2=L2,Linf=Li,Linf_boundary=bnd,Linf_interior=intr,
            argmax_x=I[kmax]*h,argmax_y=J[kmax]*h,asym_eta=eta))
for k,r in enumerate(rows):
    if k and rows[k-1]["closure"]==r["closure"] and rows[k-1]["bc"]==r["bc"]:
        for nm in["L2","Linf"]: r["p_"+nm]=np.log2(rows[k-1][nm]/r[nm])
    else: r["p_L2"]=r["p_Linf"]=np.nan
with open("ticket_d2_results.csv","w",newline="") as fh:
    wr=csv.DictWriter(fh,fieldnames=list(rows[-1])); wr.writeheader(); wr.writerows(rows)
for r in rows: print(f"{r['bc']:7s} {r['closure']:5s} N={r['N']:3d} L2={r['L2']:.3e} p={r['p_L2']:.2f} Linf={r['Linf']:.3e} p={r['p_Linf']:.2f} at({r['argmax_x']:.3f},{r['argmax_y']:.3f}) bnd={r['Linf_boundary']:.2e} int={r['Linf_interior']:.2e} eta={r['asym_eta']:.3f}")
# CG on nonsymmetric ghost Robin matrix, N=64
A,F,Ue,h,I,J=build(64,2.0,"ghost"); Ud=spla.spsolve(A.tocsc(),F)
x,info=spla.cg(A,F,rtol=1e-12,atol=0,maxiter=5000)
print(f"CG on ghost-Robin N=64: info={info}, ||x-direct||_inf={np.abs(x-Ud).max():.3e}, disc err={np.abs(Ud-Ue).max():.3e}")
# symmetrise: scale boundary rows by 1/2 -> symmetric?
D=np.where(I==64,0.5,1.0); As=sp.diags(D)@A; print("eta after halving boundary rows:",spla.norm(As-As.T)/spla.norm(As))
x2,info2=spla.cg(As,D*F,rtol=1e-12,atol=0,maxiter=5000); print(f"CG on row-scaled: info={info2}, ||x-direct||_inf={np.abs(x2-Ud).max():.3e}")
