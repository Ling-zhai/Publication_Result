"""Resumable King County spatial SHAP experiments aligned with James's manuscript.

Run independently:
  python spatial_shap_experiments.py prepare
  python spatial_shap_experiments.py shap --limit 600
  python spatial_shap_experiments.py maps
  python spatial_shap_experiments.py regional --grid-size 20
  python spatial_shap_experiments.py summary

Only writes spatial_shap_results/, never publication_results/ or publication_result_1/.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'spatial_shap_results'
CACHE = OUT / 'cache'
FIG = OUT / 'figures'
TABLE = OUT / 'tables'
FEATURES = ('grade', 'sqft_living')
K = 20
GLOBAL_BACKGROUND_SIZE = 100
SEED = 42


def dirs():
    for p in (OUT, CACHE, FIG, TABLE): p.mkdir(parents=True, exist_ok=True)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def distances(lat1, lon1, lat2, lon2):
    """Haversine distances in kilometres."""
    a, b = np.radians(np.asarray(lat1, dtype=float))[:, None], np.radians(np.asarray(lon1, dtype=float))[:, None]
    c, d = np.radians(np.asarray(lat2, dtype=float))[None, :], np.radians(np.asarray(lon2, dtype=float))[None, :]
    h = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 12742.0*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def prepare():
    dirs()
    source = ROOT / 'kc_house_data.csv'
    if not source.exists(): raise FileNotFoundError('Place kc_house_data.csv next to spatial_shap_experiments.py')
    df = pd.read_csv(source).drop(columns=['id','date'])
    sample = df.sample(n=3000, random_state=SEED).copy()
    coords = sample[['lat','long']]
    target = sample['price']
    X = sample.drop(columns=['price','lat','long','zipcode'])
    Xtr, Xte, ytr, yte = train_test_split(X, target, test_size=.2, random_state=SEED)
    ctr, cte = coords.loc[Xtr.index], coords.loc[Xte.index]
    missing = sorted(set(FEATURES)-set(X.columns))
    if missing: raise ValueError('Missing features: '+str(missing))
    model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(Xtr, ytr)
    # Keep the background exactly the same for global SHAP and global mean prediction.
    bg = Xtr.sample(n=GLOBAL_BACKGROUND_SIZE, random_state=SEED)
    global_baseline = float(np.mean(model.predict(bg)))
    payload = dict(model=model, X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
                   coord_train=ctr, coord_test=cte, global_background=bg,
                   global_baseline=global_baseline, feature_names=list(X.columns))
    joblib.dump(payload, CACHE/'prepared.joblib', compress=3)
    pred = model.predict(Xte)
    info = dict(train_count=len(Xtr), test_count=len(Xte), features=list(X.columns),
                global_baseline=global_baseline, r2=float(r2_score(yte,pred)),
                rmse=float(np.sqrt(mean_squared_error(yte,pred))),
                mae=float(mean_absolute_error(yte,pred)), selected_features=list(FEATURES), K=K)
    save_json(TABLE/'model_metrics.json', info)
    print('Prepared and cached model. Model metrics:', info)


def get():
    path=CACHE/'prepared.joblib'
    if not path.exists(): raise FileNotFoundError('First run: python spatial_shap_experiments.py prepare')
    return joblib.load(path)


def shap_run(limit, chunk_size):
    dirs(); d=get()
    Xte=d['X_test'].iloc[:limit]
    ctr, cte=d['coord_train'], d['coord_test'].loc[Xte.index]
    neighbors=np.argsort(distances(cte.lat, cte.long, ctr.lat, ctr.long), axis=1)[:,:K]
    model=d['model']
    gl_exp=shap.TreeExplainer(model, data=d['global_background'],feature_perturbation='interventional')
    global_base=float(np.ravel(gl_exp.expected_value)[0])
    assert abs(global_base-d['global_baseline'])<0.1, 'Global baseline inconsistent'
    for start in range(0,len(Xte),chunk_size):
        end=min(start+chunk_size,len(Xte))
        fn=CACHE/f'shap_{start:04d}_{end:04d}.npz'
        if fn.exists(): print('Reuse cached',fn.name);continue
        Xpart=Xte.iloc[start:end]
        std=np.asarray(gl_exp.shap_values(Xpart,check_additivity=False))
        loc=np.zeros_like(std,dtype=float)
        loc_base=np.zeros(len(Xpart))
        for q in range(len(Xpart)):
            row=start+q
            bg=d['X_train'].iloc[neighbors[row]]
            ex=shap.TreeExplainer(model,data=bg,feature_perturbation='interventional')
            loc[q]=np.asarray(ex.shap_values(Xpart.iloc[[q]],check_additivity=False))[0]
            loc_base[q]=float(np.ravel(ex.expected_value)[0])
        pred=model.predict(Xpart)
        # Numerical checks are saved; a failed decomposition should never be silently used.
        err_gl=pred-(global_base+std.sum(axis=1))
        err_loc=pred-(loc_base+loc.sum(axis=1))
        if max(np.abs(err_gl).max(),np.abs(err_loc).max())>1.0:
            raise RuntimeError(f'SHAP additivity error > $1 in batch {start}:{end}')
        np.savez_compressed(fn, indices=Xpart.index.to_numpy(),standard=std,local=loc,
                            global_base=np.full(len(Xpart),global_base),local_base=loc_base,
                            predictions=pred,global_error=err_gl,local_error=err_loc)
        print(f'Cached SHAP {end}/{len(Xte)} houses')
    print('Finished. Now run maps and summary separately.')


def load_shap():
    d=get()
    files=sorted(CACHE.glob('shap_????_????.npz'))
    if not files: raise FileNotFoundError('First run: python spatial_shap_experiments.py shap --limit 600')
    blocks=[];indices=[]
    # Strictly require a contiguous sequence; do not mix partial runs accidentally.
    expected=0
    for fn in files:
        start,end=map(int,fn.stem.split('_')[1:])
        if start!=expected: raise RuntimeError('SHAP cache has gaps / overlapping ranges; remove stale chunks or use one chunk size')
        z=np.load(fn)
        if end-start!=len(z['indices']): raise RuntimeError('Bad SHAP cache '+fn.name)
        blocks.append({key:z[key] for key in z.files});indices.extend(z['indices'].tolist());expected=end
    if len(indices)>len(d['X_test']) or indices!=d['X_test'].index[:len(indices)].tolist():
        raise RuntimeError('SHAP cache does not correspond to current test sample; clear cache before rerunning prepare')
    return d,pd.Index(indices),{key:np.concatenate([b[key] for b in blocks],axis=0) for key in blocks[0] if key!='indices'}


def plot_spatial(lon,lat,values,title,label,path,diverging=True):
    plt.figure(figsize=(9,7))
    if diverging:
        vmax=float(np.max(np.abs(values))) or 1
        sc=plt.scatter(lon,lat,c=values/1000,cmap='coolwarm',vmin=-vmax/1000,vmax=vmax/1000,s=20)
    else:
        sc=plt.scatter(lon,lat,c=values/1000,cmap='viridis',s=20)
    plt.colorbar(sc,label=label+' ($1,000s)')
    plt.xlabel('Longitude');plt.ylabel('Latitude');plt.title(title)
    plt.tight_layout();plt.savefig(path,dpi=250);plt.close()


def maps():
    dirs();d,ind,z=load_shap()
    loc=d['coord_test'].loc[ind]
    L=z['local_base']-z['global_base']
    # Identity f(x)-mu = L(u)+sum_j spatial_j, with matched baselines.
    identity=(z['standard'].sum(axis=1)-L-z['local'].sum(axis=1))
    stats={'houses':len(ind),'max_decomposition_difference_usd':float(np.max(np.abs(identity))),
           'mean_absolute_location_effect_usd':float(np.mean(np.abs(L)))}
    save_json(TABLE/'decomposition_check.json',stats)
    if stats['max_decomposition_difference_usd']>1.0: raise RuntimeError('Decomposition failed: '+str(stats))
    frame=pd.DataFrame({'index':ind,'lat':loc.lat.to_numpy(),'long':loc.long.to_numpy(),
                        'location_effect_usd':L,'global_baseline_usd':z['global_base'],
                        'local_baseline_usd':z['local_base'],'prediction_usd':z['predictions']})
    frame.to_csv(TABLE/'location_effect.csv',index=False)
    plot_spatial(frame.long,frame.lat,L,'Overall location effect', 'Local minus global baseline',FIG/'overall_location_effect.png')
    for feature in FEATURES:
        j=d['feature_names'].index(feature)
        std=z['standard'][:,j];local=z['local'][:,j];delta=local-std
        t=frame[['index','lat','long']].copy()
        t['standard_shap_usd']=std;t['spatial_shap_usd']=local;t['spatial_minus_standard_usd']=delta
        t.to_csv(TABLE/f'{feature}_shap.csv',index=False)
        plot_spatial(t.long,t.lat,std,f'Standard SHAP: {feature}','Standard SHAP',FIG/f'{feature}_standard_shap.png')
        plot_spatial(t.long,t.lat,local,f'KNN Spatial SHAP: {feature}, K={K}','Spatial SHAP',FIG/f'{feature}_spatial_shap.png')
        plot_spatial(t.long,t.lat,delta,f'Spatial minus standard: {feature}','Difference',FIG/f'{feature}_shap_difference.png')
    print('Maps saved in',FIG)


def regional(grid_size, chunk_size, n_eval, max_distance_km):
    dirs();d=get();Xtr=d['X_train'];ctr=d['coord_train'];model=d['model']
    # Define evaluation samples from local training observations, matching P_u in James's Eq. for I_i(u).
    # For test-only evaluation replace with test samples, but then the sampling distribution is different.
    lats=np.linspace(ctr.lat.min(),ctr.lat.max(),grid_size)
    lons=np.linspace(ctr.long.min(),ctr.long.max(),grid_size)
    lg,tg=np.meshgrid(lons,lats)
    latitude=tg.ravel();longitude=lg.ravel()
    dst=distances(latitude,longitude,ctr.lat,ctr.long)
    neighbor=np.argsort(dst,axis=1)[:,:max(K,n_eval)]
    nearest=np.min(dst,axis=1)
    pfx=f'regional_g{grid_size}_eval{n_eval}_k{K}'
    for start in range(0,len(latitude),chunk_size):
        end=min(start+chunk_size,len(latitude))
        fn=CACHE/f'{pfx}_{start:04d}_{end:04d}.csv'
        if fn.exists(): print('Reuse',fn.name);continue
        results=[]
        for r in range(start,end):
            rec={'grid_index':r,'lat':latitude[r],'long':longitude[r],
                 'nearest_training_km':nearest[r]}
            if nearest[r]>max_distance_km:
                for f in FEATURES: rec[f+'_importance_usd']=np.nan
            else:
                bg=Xtr.iloc[neighbor[r,:K]]
                sample=Xtr.iloc[neighbor[r,:n_eval]]
                exp=shap.TreeExplainer(model,data=bg,feature_perturbation='interventional')
                vals=np.asarray(exp.shap_values(sample,check_additivity=False))
                for f in FEATURES:
                    rec[f+'_importance_usd']=float(np.mean(np.abs(vals[:,d['feature_names'].index(f)])))
            results.append(rec)
        pd.DataFrame(results).to_csv(fn,index=False)
        print('Regional cells cached',end,'/',len(latitude))
    files=sorted(CACHE.glob(pfx+'_????_????.csv'))
    parts=[pd.read_csv(p) for p in files]
    full=pd.concat(parts,ignore_index=True).sort_values('grid_index').drop_duplicates('grid_index')
    if full.grid_index.tolist()!=list(range(len(latitude))):
        raise RuntimeError('Regional grid incomplete; rerun with same grid and chunk settings')
    full.to_csv(TABLE/f'{pfx}.csv',index=False)
    for f in FEATURES:
        zz=full[f+'_importance_usd'].to_numpy().reshape(grid_size,grid_size)/1000
        fig,ax=plt.subplots(figsize=(9,7))
        im=ax.pcolormesh(lg,tg,np.ma.masked_invalid(zz),cmap='viridis',shading='auto')
        fig.colorbar(im,ax=ax,label='Mean absolute Spatial SHAP ($1,000s)')
        ax.set(xlabel='Longitude',ylabel='Latitude',title=f'Regional importance: {f} (masked > {max_distance_km} km)')
        fig.tight_layout();fig.savefig(FIG/f'{f}_regional_importance.png',dpi=250);plt.close(fig)
    print('Regional results saved; cells farther than threshold are masked.')


def summary():
    dirs();d,ind,z=load_shap()
    L=z['local_base']-z['global_base']
    rows=[]
    for feature in FEATURES:
        idx=d['feature_names'].index(feature)
        a=z['standard'][:,idx];b=z['local'][:,idx]
        rows.append({'feature':feature,'houses':len(ind),'mean_absolute_difference_usd':np.mean(np.abs(a-b)),
                     'sign_change_percent':100*np.mean(np.sign(a)!=np.sign(b)),
                     'pearson_r':np.corrcoef(a,b)[0,1]})
    pd.DataFrame(rows).to_csv(TABLE/'feature_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('Mean absolute location effect (USD):',np.mean(np.abs(L)))
    print('Max standard additivity error (USD):',np.max(np.abs(z['global_error'])))
    print('Max local additivity error (USD):',np.max(np.abs(z['local_error'])))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='step',required=True)
    sub.add_parser('prepare')
    ps=sub.add_parser('shap');ps.add_argument('--limit',type=int,default=600);ps.add_argument('--chunk-size',type=int,default=25)
    sub.add_parser('maps')
    pr=sub.add_parser('regional');pr.add_argument('--grid-size',type=int,default=20)
    pr.add_argument('--chunk-size',type=int,default=20);pr.add_argument('--n-eval',type=int,default=10)
    pr.add_argument('--max-distance-km',type=float,default=10)
    sub.add_parser('summary')
    args=p.parse_args()
    if args.step=='prepare':prepare()
    elif args.step=='shap':shap_run(args.limit,args.chunk_size)
    elif args.step=='maps':maps()
    elif args.step=='regional':regional(args.grid_size,args.chunk_size,args.n_eval,args.max_distance_km)
    else:summary()


if __name__=='__main__':main()
