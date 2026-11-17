"""Shot-based, multi-expiry LCHS calibration with an explicit circuit boundary.

Only counts returned by ``runner.run`` enter the measured optimiser. Dense
small-system Hamiltonian unitaries are compiled, and their classical synthesis
cost is recorded. This is an executable prototype, not an efficient-oracle claim.
"""
import os, sys, time, hashlib, platform, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm, svd
from scipy.stats import beta as beta_distribution
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from surface_model import (grid, data_for, generator, weights, prices, loss,
                           financial_metrics, save_json, TIMES, TRUTH)

def dilation(B):
    u,s,vh=svd(B); z=np.sqrt(np.maximum(0,1-s*s)); v=vh.conj().T
    R=np.block([[B,(u*z)@u.conj().T],[(v*z)@v.conj().T,-B.conj().T]])
    if not np.allclose(R.conj().T@R,np.eye(len(R)),atol=1e-11):
        raise AssertionError('Nonunitary observation dilation')
    return R

def build_residual(problem, xi, expiry, terms=32, cutoff=32.):
    """Residual through expiry inclusive, retaining distinct LCU workspaces.

    Input coefficients are classical. Later expiries evolve from the original
    payoff through EVERY preceding interval, without reading intermediate states.
    """
    g=problem['g']; N=g['N']; n=g['n']; S=expiry+1
    b=int(np.log2(terms)); assert 2**b==terms
    xi=np.asarray(xi).reshape(len(problem['times']),g['M'])
    k,w,meta=weights(terms,cutoff); alpha=meta['alpha']
    work=list(range(n)); kernels=[list(range(n+s*b,n+(s+1)*b)) for s in range(S)]
    wa=n+S*b; branch=wa+1; width=branch+1
    obs=problem['observations'][expiry]; W0=obs[:,1:-1]; Q=len(obs)
    if Q>N: raise ValueError('Small prototype requires number of quotes <= grid size')
    W=np.zeros((N,N)); W[:Q]=W0
    y=np.zeros(N); y[:Q]=problem['quotes'][expiry]-obs@g['G']
    gamma=max(1.,np.linalg.norm(W,2)); unorm=np.linalg.norm(g['v0']); ynorm=np.linalg.norm(y)
    T=problem['times'][expiry]
    a=unorm*np.exp(g['shift']*T)*alpha**S*gamma; scale=a+ynorm
    prep=StatePreparation(np.sqrt(abs(w)/alpha))
    forward=QuantumCircuit(width-1,name=f'PDE-T{expiry+1}')
    forward.append(StatePreparation(g['v0']/unorm),work)
    last=0.
    for s in range(S):
        dt=problem['times'][s]-last; last=problem['times'][s]
        A=generator(g,xi[s]); H=-.5j*(A-A.T); L=(A+A.T)/2+g['shift']*np.eye(N)
        kr=kernels[s]; forward.append(prep,kr)
        for j in range(terms):
            U=np.exp(1j*np.angle(w[j]))*expm(-1j*dt*(H+k[j]*L))
            # There is no enumeration of coefficient designs or preloaded loss.
            gate=UnitaryGate(U,label=f'U{s},{j}').control(b,ctrl_state=j)
            forward.append(gate,kr+work)
        forward.append(prep.inverse(),kr)
    forward.append(UnitaryGate(dilation(W/gamma),label='Quote map'),work+[wa])
    V=QuantumCircuit(width,name=f'Residual-T{expiry+1}')
    mix=StatePreparation([np.sqrt(a/scale),np.sqrt(ynorm/scale)])
    V.append(mix,[branch]); V.x(branch)
    V.append(forward.to_gate().control(1),[branch]+list(range(width-1)))
    V.x(branch)
    V.append(StatePreparation(y/ynorm).control(1),[branch]+work)
    V.z(branch); V.append(mix.inverse(),[branch])
    anc=[q for kr in kernels for q in kr]+[wa,branch]
    return V,dict(expiry=expiry,quotes=Q,scale=scale,scale_to_mse=scale**2/Q,
                  ancillas=anc,qubits=width,terms=terms,cutoff=cutoff,alpha=alpha,
                  price_amplification=float(np.exp(g['shift']*T)*alpha**S))

def compile_residual(problem,xi,expiry,terms,cutoff):
    start=time.perf_counter(); V,meta=build_residual(problem,xi,expiry,terms,cutoff)
    V.add_register(__import__('qiskit').ClassicalRegister(len(meta['ancillas']),'success'))
    V.measure(meta['ancillas'],range(len(meta['ancillas'])))
    qc=transpile(V,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=17)
    meta.update(cx=int(qc.count_ops().get('cx',0)),u=int(qc.count_ops().get('u',0)),
                depth=qc.depth(),compile_seconds=time.perf_counter()-start)
    return qc,meta

class MeasuredObjective:
    """Inject an AerSimulator or a compatible Qiskit backend runner.

    A real backend requires target-aware transpilation and suitable logical
    resources. No job is sent to any external backend by the default notebook.
    """
    def __init__(self,problem,shots=262144,seed=101,terms=32,cutoff=32.,runner=None,cache=None):
        self.problem=problem; self.shots=shots; self.seed=seed
        self.terms=terms; self.cutoff=cutoff
        self.runner=runner or AerSimulator(method='statevector',max_parallel_threads=1)
        self.cache={} if cache is None else cache
        self.records=[]; self.call=0
    def __call__(self,xi):
        self.call+=1; circuits=[]; metadata=[]
        xi=np.asarray(xi).reshape(len(self.problem['times']),self.problem['g']['M'])
        for s in range(len(self.problem['times'])):
            # Cache only circuits for an identical parameter vector, never losses.
            key=(s,tuple(xi[:s+1].ravel()),self.terms,self.cutoff)
            if key not in self.cache:
                self.cache[key]=compile_residual(self.problem,xi,s,self.terms,self.cutoff)
            qc,meta=self.cache[key]; circuits.append(qc); metadata.append(meta)
        started=time.perf_counter()
        run_options={'shots':self.shots}
        if isinstance(self.runner,AerSimulator):
            run_options['seed_simulator']=self.seed+1009*self.call
        result=self.runner.run(circuits,**run_options).result()
        elapsed=time.perf_counter()-started; estimates=[]; variances=[]
        for s,meta in enumerate(metadata):
            counts=result.get_counts(s); hits=counts.get('0'*len(meta['ancillas']),0)
            p=hits/self.shots; fac=meta['scale_to_mse']
            tail=.05/(2*len(metadata))
            lower=0. if hits==0 else beta_distribution.ppf(tail,hits,self.shots-hits+1)
            upper=1. if hits==self.shots else beta_distribution.ppf(1-tail,hits+1,self.shots-hits)
            estimates.append(fac*p); variances.append(fac**2*p*(1-p)/self.shots)
            self.records.append(dict(call=self.call,expiry=s+1,shots=self.shots,hits=hits,
                probability=p,mse_estimate=fac*p,mse_standard_error=np.sqrt(variances[-1]),
                mse_lower=fac*lower,mse_upper=fac*upper,
                sampler_seed=self.seed,parameters=xi.ravel().tolist(),
                cx_per_execution=meta['cx'],depth=meta['depth'],qubits=meta['qubits'],
                backend_seconds=elapsed/len(metadata),compile_seconds=meta['compile_seconds']))
        return float(np.mean(estimates))

def coordinate_search(objective,shape=(2,2),rounds=2,step=.25):
    """Identical bounded pattern search for measured and matrix objectives.

    The incumbent is remeasured for every search direction. All candidates have the
    same shot budget. No exact prices, gradients or validation scores are used.
    """
    xi=np.full(shape,.5); history=[]; evaluation=0
    directions=list(np.eye(xi.size))
    # Opposing within-expiry changes directly explore smile tilt as well as level.
    for s in range(shape[0]):
        d=np.zeros(xi.size);d[s*shape[1]]=1;d[s*shape[1]+1]=-1;directions.append(d)
    for sweep in range(rounds):
        for j,direction in enumerate(directions):
            trials=[]
            candidates=[xi.copy()]
            for sign in [-1,1]:
                x=np.clip(xi.ravel()+sign*step*direction,0,1).reshape(shape)
                if not any(np.array_equal(x,y) for y in candidates):candidates.append(x)
            for x in candidates:
                f=objective(x);evaluation+=1
                trials.append((f,x));history.append(dict(sweep=sweep,coordinate=j,evaluation=evaluation,
                    loss=f,parameters=x.ravel().tolist()))
            _,xi=min(trials,key=lambda z:z[0])
    return xi,pd.DataFrame(history)

def circuit_checks(problem,terms=8,cutoff=4.,out=None):
    rows=[]
    for xi in [np.full((2,2),.5),TRUTH]:
        p=prices(problem['g'],xi,method='lchs',terms=terms,cutoff=cutoff)
        for s in range(2):
            V,meta=build_residual(problem,xi,s,terms,cutoff)
            state=Statevector.from_instruction(V).data
            # Work qubits are low bits, all successful ancillary bits are zero.
            observed=float(np.sum(abs(state[:problem['g']['N']])**2))
            expected=float(np.mean(abs(problem['observations'][s]@p[s]-problem['quotes'][s])**2)/meta['scale_to_mse'])
            error=abs(observed-expected)
            if error>2e-10:raise AssertionError(f'Multi-expiry residual mismatch: {error}')
            rows.append(dict(expiry=s+1,parameters=xi.ravel().tolist(),
                             probability=observed,expected=expected,absolute_error=error))
    frame=pd.DataFrame(rows)
    if out:frame.to_csv(Path(out)/'circuit_validation.csv',index=False)
    return frame

def run_calibration_experiment(out,profile='paper',runner=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    configs={'smoke':dict(terms=4,cutoff=4.,shots=[4096],seeds=[101],rounds=1),
             'pilot':dict(terms=8,cutoff=12.,shots=[65536,262144],seeds=[101,202],rounds=1),
             'paper':dict(terms=8,cutoff=12.,shots=[262144,4194304],seeds=[101,202,303],rounds=2),
             'precision':dict(terms=32,cutoff=32.,shots=[1048576,16777216],seeds=[101,202,303],rounds=2)}
    cfg=configs[profile];problem=data_for(grid());cache={}
    save_json(out/'configuration.json',dict(version='2026-09-09-r2',profile=profile,**cfg,
       target='same-grid synthetic exact PDE',truth=TRUTH,quotes=problem['quotes'],
       grid=problem['g']['K'],times=TIMES,noise_bp=0,hardware_execution=False,
       implementation='coherent LCHS within each maturity; measured residual; classical bounded pattern search',
       synthesis='dense small-system Hamiltonian unitaries; all-to-all u/cx',
       optimiser_access='measurement counts only; no statevector, exact objective or gradient'))
    save_json(out/'environment.json',dict(python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','qiskit','qiskit-aer']},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__),Path(__file__).with_name('surface_model.py')]},
        sampling='Aer finite-shot counts from decomposed circuits',
        seed_policy='base sampler seed + 1009 * objective call index'))
    circuit_checks(problem,min(cfg['terms'],8),min(cfg['cutoff'],12),out)
    results=[];histories=[];measurements=[]
    for method in ['exact','fd','lchs']:
        start=time.perf_counter()
        x,h=coordinate_search(lambda z:loss(problem,z,method,cfg['terms'],cfg['cutoff']),rounds=cfg['rounds'])
        elapsed=time.perf_counter()-start
        row=dict(method=method,shots=0,seed=0,parameters=x.ravel().tolist(),evaluations=len(h),cpu_seconds=elapsed,
                 **financial_metrics(problem,x,method,cfg['terms'],cfg['cutoff']))
        results.append(row);h['method']=method;h['shots']=0;h['seed']=0;histories.append(h)
    for shots in cfg['shots']:
        for seed in cfg['seeds']:
            print(f'Measured calibration: shots={shots}, seed={seed}',flush=True)
            obj=MeasuredObjective(problem,shots,seed,cfg['terms'],cfg['cutoff'],runner,cache)
            start=time.perf_counter();x,h=coordinate_search(obj,rounds=cfg['rounds']);elapsed=time.perf_counter()-start
            row=dict(method='quantum_counts',shots=shots,seed=seed,parameters=x.ravel().tolist(),evaluations=len(h),
                     cpu_seconds=elapsed,**financial_metrics(problem,x,'lchs',cfg['terms'],cfg['cutoff']))
            row['total_shots']=sum(r['shots'] for r in obj.records)
            row['total_cx_executions']=sum(r['shots']*r['cx_per_execution'] for r in obj.records)
            row['max_qubits']=max(r['qubits'] for r in obj.records)
            results.append(row);h['method']='quantum_counts';h['shots']=shots;h['seed']=seed;histories.append(h)
            # Independent final evaluation is not used to select the coefficients.
            final_start=len(obj.records);row['independent_final_measured_loss']=obj(x)
            final_records=obj.records[final_start:]
            row['independent_final_mse_lower']=float(np.mean([r['mse_lower'] for r in final_records]))
            row['independent_final_mse_upper']=float(np.mean([r['mse_upper'] for r in final_records]))
            for record in obj.records:
                record['purpose']='calibration' if record['call']<=len(h) else 'independent_final_check'
            measurements.extend(obj.records)
            row['verification_shots']=len(final_records)*shots
            row['verification_cx_executions']=sum(r['shots']*r['cx_per_execution'] for r in final_records)
            # Incremental checkpoints retain completed runs if Colab disconnects.
            pd.DataFrame(results).to_csv(out/'calibration_results.csv',index=False)
            pd.concat(histories,ignore_index=True).to_csv(out/'optimisation_history.csv',index=False)
            pd.DataFrame(measurements).to_json(out/'measurement_log.json',orient='records',indent=2)
            save_json(out/'checkpoint.json',dict(completed=len(results)-3,planned=len(cfg['shots'])*len(cfg['seeds']),last=row))
    # Export actual compiled circuits for the final measured coefficients.
    circuits=[];resource=[]
    for s in range(2):
        qc,meta=compile_residual(problem,x,s,cfg['terms'],cfg['cutoff'])
        circuits.append(qc);resource.append(meta)
    with (out/'final_residual_circuits.qpy').open('wb') as f:qpy.dump(circuits,f)
    save_json(out/'final_circuit_resources.json',resource)
    save_json(out/'status.json',dict(completed=True,profile=profile,hardware_execution=False,
        measured_runs=len(cfg['shots'])*len(cfg['seeds']),no_statevector_access_in_optimiser=True))
    return pd.DataFrame(results)

def resource_sweep(out,profile='paper'):
    out=Path(out); rows=[]
    cases=[(2,4,1),(2,4,2),(2,8,1),(2,8,2),(3,4,1),(3,4,2)]
    if profile=='paper':cases += [(2,16,1),(2,16,2),(2,32,1),(2,32,2),(3,8,1),(3,8,2)]
    for n,J,S in cases:
        p=data_for(grid(n));x=np.full((2,2),.5)
        cutoff=4. if J<=8 else 32.
        _,m=compile_residual(p,x,S-1,J,cutoff)
        rows.append(dict(N=2**n,J=J,cutoff=cutoff,expiries=S,**{k:m[k] for k in ['qubits','cx','u','depth','compile_seconds','price_amplification']}))
        print(f'Resource circuit N={2**n}, J={J}, expiries={S}: {m["qubits"]} qubits',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'resource_sweep.csv',index=False);return df
