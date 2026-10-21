"""Discover customer groups and evaluate clustering stability using independent fits."""
from kit import *
TITLE='Customer Segmentation with K-Means++'
LIVE=False

def fit(x,k,seed=42,steps=60):
    rng=np.random.default_rng(seed);centers=[x[rng.integers(len(x))]]
    for _ in range(k-1):
        dist=np.min(np.sum((x[:,None]-np.array(centers))**2,axis=2),axis=1)
        centers.append(x[rng.choice(len(x),p=dist/dist.sum())] if dist.sum()>0 else x[rng.integers(len(x))])
    c=np.array(centers)
    for _ in range(steps):
        labels=assign(x,c);new=np.array([x[labels==i].mean(axis=0) if np.any(labels==i) else c[i] for i in range(k)])
        if np.allclose(c,new):break
        c=new
    return c
def assign(x,c):return np.argmin(np.sum((x[:,None]-c)**2,axis=2),axis=1)
def silhouette(x,labels):
    dist=np.sqrt(np.sum((x[:,None]-x[None])**2,axis=2));values=[]
    for i,label in enumerate(labels):
        same=np.where(labels==label)[0];same=same[same!=i]
        if not len(same):values.append(0.);continue
        aa=dist[i,same].mean();bb=min((dist[i,labels==other].mean() for other in np.unique(labels) if other!=label),default=0.)
        values.append((bb-aa)/max(aa,bb,1e-12))
    return float(np.mean(values))
def ari(y,p):
    table=np.zeros((len(np.unique(y)),len(np.unique(p))),int)
    for a,b in zip(y,p):table[int(a),int(b)]+=1
    choose=lambda v:np.sum(v*(v-1)/2)
    n=choose(table);row=choose(table.sum(axis=1));col=choose(table.sum(axis=0));expected=row*col/(len(y)*(len(y)-1)/2)
    return float((n-expected)/max((row+col)/2-expected,1e-12))
def run(d,a):
    rng=np.random.default_rng(a.seed);labels=rng.integers(0,3,600);means=np.array([[-4,0],[0,4],[4,-1]]);x=means[labels]+rng.normal(0,.6,(600,2));mean=x[:400].mean(axis=0);std=x[:400].std(axis=0);z=(x-mean)/std
    candidates=[(k,fit(z[:400],k,a.seed)) for k in (2,3,4,5)];k,c=max(candidates,key=lambda pair:silhouette(z[400:],assign(z[400:],pair[1])))
    pred=assign(z[400:],c)
    return dict(selected_k=k,validation_silhouette=silhouette(z[400:],pred),external_ari=ari(labels[400:],pred),profiles=(c*std+mean).tolist(),caveat='The 200 held-out rows select k, so silhouette is validation performance, not an unbiased final test.'),dict(centers=c,mean=mean,std=std,x=x)

if __name__=="__main__":main(run)
